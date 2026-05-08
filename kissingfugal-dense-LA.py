import networkx as nx
import math
import torch
import torch.nn.functional as F
import numpy as np
from helpers.pred import feature_extraction, convertToPermHungarian
import scipy
import csv
import itertools
import time

# path = "/home/cheng/Fugal/data/real_noise/ACM-DBLP/pos_pairs.npy"
# data = np.load(path)
# ground_truth = {pair[0]: pair[1] for pair in data}
# # print("Ground truth mapping: ", ground_truth)

import math

class FAVORPlusMapping:
    def __init__(self, input_dim, nb_features, device, seed=42):
        self.input_dim = input_dim
        self.nb_features = nb_features
        self.device = device
        
        # 1. 生成正交随机矩阵 (Orthogonal Random Matrix)
        # 我们先生成正态分布矩阵，然后通过 QR 分解使其正交
        nb_blocks = math.ceil(nb_features / input_dim)
        W_list = []
        for _ in range(nb_blocks):
            q, _ = torch.linalg.qr(torch.randn(input_dim, input_dim, device=device))
            W_list.append(q.T)
            
        # 拼接并截取到所需的 nb_features 维度 [input_dim, nb_features]
        W_ortho = torch.cat(W_list, dim=0)[:nb_features].T
        
        # 2. 对每一行进行模长重采样（符合奇异值分布，模拟高斯核）
        multiplier = torch.randn((input_dim, nb_features), device=device).norm(dim=0)
        self.W = W_ortho * multiplier
        
    def map(self, x):
        """
        x: [n, input_dim]
        returns: [n, nb_features]
        """
        # 计算内积 x @ W
        projection = x @ self.W
        
        # FAVOR+ 的正向映射公式 (针对 Softmax/Gaussian 近似):
        # phi(x) = exp(-||x||^2 / 2) * exp(W^T x)
        # 为了数值稳定性，我们通常使用具有正向保证的映射
        x_norm_squared = torch.sum(x**2, dim=-1, keepdim=True)
        
        # 使用偏置和指数函数模拟
        # 加上常数项和归一化系数
        diag_data = -0.5 * x_norm_squared
        # 这里是 Positive Random Features 的核心逻辑
        phi = torch.exp(projection + diag_data) / math.sqrt(self.nb_features)
        
        return phi


def read_file():
    query_path = "/home/cheng/fly/data/real_noise/MultiMagna/yeast0_Y2H1.txt"
    target_path = "/home/cheng/fly/data/real_noise/MultiMagna/yeast5_Y2H1.txt"

    n = 1004
    Gq = nx.Graph()
    Gt = nx.Graph()

    for i in range(n):
        Gq.add_node(i)
    for i in range(n):
        Gt.add_node(i)

    with open(query_path) as f:
        for line in f:
            u, v = map(int, line.strip().split())
            Gq.add_edge(u, v)
            
    with open(target_path) as f:        
        for line in f:
            u, v = map(int, line.strip().split())
            Gt.add_edge(u, v)

    return Gq, Gt, n


def build_inputs(Gq: nx.Graph, Gt: nx.Graph, dtype=torch.float32):
    A = torch.tensor(nx.to_numpy_array(Gq), dtype=dtype)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype=dtype)

    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)

    if not isinstance(F1, torch.Tensor):
        F1 = torch.tensor(F1, dtype=dtype)
    else:
        F1 = F1.to(dtype=dtype)

    if not isinstance(F2, torch.Tensor):
        F2 = torch.tensor(F2, dtype=dtype)
    else:
        F2 = F2.to(dtype=dtype)

    D = torch.cdist(F1, F2, p=2)
    return A, B, D


def kernel_feature(X, kind="elu", favor_mapper=None):
    # print(f"Using kernel feature: {kind}")
    if kind == "elu":
        return F.elu(X) + 1.0
    elif kind == "softplus":
        return F.softplus(X) + 1e-6
    elif kind == "exp":
        return torch.exp(X.clamp(min=-10, max=10))
    elif kind == "poly2":
        return torch.cat([F.relu(X), torch.pow(F.relu(X), 2)], dim=-1)
    elif kind == "favor+":
        if favor_mapper is None:
            raise ValueError("favor_mapper must be provided when kind='favor+'")
        return favor_mapper.map(X)
    elif kind == "relu_squared":
        return torch.pow(F.relu(X), 2) + 1e-6
    else:
        raise ValueError(kind)


def linear_matching_factors(
    V: torch.Tensor,
    W: torch.Tensor,
    beta: float = 20.0,
    kind: str = "elu",
    favor_mapper=None,
):
    """Return factors for P ~= R @ K.T without materializing the n x n matrix."""
    Vn = V / torch.linalg.norm(V, dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / torch.linalg.norm(W, dim=1, keepdim=True).clamp_min(1e-12)
    # Vn = V
    # Wn = W

    scale = math.sqrt(beta)
    Q = kernel_feature(scale * Vn, kind=kind, favor_mapper=favor_mapper)
    K = kernel_feature(scale * Wn, kind=kind, favor_mapper=favor_mapper)
    denom = (Q @ K.sum(dim=0)).clamp_min(1e-12)
    R = Q / denom[:, None]
    return R, K


def make_soft_matching(
    V: torch.Tensor,
    W: torch.Tensor,
    beta: float = 20.0,
    kind: str = "elu",
    favor_mapper=None,
) -> torch.Tensor:
    """Build P with a linear-attention kernel form.

    P_ij ~= phi(sqrt(beta) V_i)^T phi(sqrt(beta) W_j)
            / sum_l phi(sqrt(beta) V_i)^T phi(sqrt(beta) W_l)
    """
    R, K = linear_matching_factors(
        V,
        W,
        beta=beta,
        kind=kind,
        favor_mapper=favor_mapper,
    )
    return R @ K.T



def linear_fugal_loss_terms(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    beta: float,
    mu: float,
    row_penalty: float,
    col_penalty: float,
    kind: str = "elu",
    favor_mapper=None,
):
    R, K = linear_matching_factors(
        V,
        W,
        beta=beta,
        kind=kind,
        favor_mapper=favor_mapper,
    )

    left = R.T @ (A @ R)
    right = K.T @ (B.T @ K)
    structure_term = -torch.trace(left @ right)

    feature_term = mu * torch.sum(R * (D @ K))

    col_sums = K @ R.sum(dim=0)
    col_constraint = torch.sum((col_sums - 1.0) ** 2)
    constraint_term =  col_penalty * col_constraint

    loss = structure_term + feature_term + constraint_term
    return loss, structure_term, feature_term, constraint_term


def train_with_adam(
    Gq: nx.Graph,
    Gt: nx.Graph,
    embed_dim: int = 30,
    beta: float = 10.0,
    mu: float = 1.0,
    row_penalty: float = 10.0,
    col_penalty: float = 10.0,
    learning_rate: float = 1e-2,
    max_iter: int = 10000,
    use_GPU: bool = True,
    kind: str = "elu",
    favor_features: int | None = None,
):
    n = Gq.number_of_nodes()
    if Gt.number_of_nodes() != n:
        raise ValueError("This prototype assumes Gq and Gt have the same number of nodes.")
    
    device = torch.device("cuda" if use_GPU and torch.cuda.is_available() else "cpu")
    # dtype = torch.float32
    dtype = torch.float32
    
    A, B, D = build_inputs(Gq, Gt)

    A = A.to(device,dtype=dtype)
    B = B.to(device,dtype=dtype)
    D = D.to(device,dtype=dtype)
    V = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
    W = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
    favor_mapper = None
    if kind == "favor+":
        favor_features = favor_features or embed_dim
        favor_mapper = FAVORPlusMapping(embed_dim, favor_features, device=device)

    with torch.no_grad():
        R_debug, K_debug = linear_matching_factors(
            V,
            W,
            beta=beta,
            kind=kind,
            favor_mapper=favor_mapper,
        )
        left_debug = R_debug.T @ (A @ R_debug)
        right_debug = K_debug.T @ (B.T @ K_debug)
        print(
            "dimensions "
            f"A={tuple(A.shape)} "
            f"B={tuple(B.shape)} "
            f"D={tuple(D.shape)} "
            f"V={tuple(V.shape)} "
            f"W={tuple(W.shape)} "
            f"R={tuple(R_debug.shape)} "
            f"K={tuple(K_debug.shape)} "
            f"left={tuple(left_debug.shape)} "
            f"right={tuple(right_debug.shape)}"
        )
        if favor_mapper is not None:
            print(f"dimensions favor_mapper.W={tuple(favor_mapper.W.shape)}")


    optimizer = torch.optim.Adam([V, W], lr=learning_rate)
    history = []

    best_loss = float("inf")
    best_V = V.detach().clone()
    best_W = W.detach().clone()
    wait = 0
    patience = max_iter
    min_delta = 1e-4

    start_time = time.time()

    for step in range(max_iter):
        loss, structure_term, feature_term, constraint_term = linear_fugal_loss_terms(
            A,
            B,
            D,
            V,
            W,
            beta=beta,
            mu=mu,
            row_penalty=row_penalty,
            col_penalty=col_penalty,
            kind=kind,
            favor_mapper=favor_mapper,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(float(loss.detach()))

        loss_value = float(loss.detach())
        # print(step, loss_value)

        if loss_value < best_loss - min_delta:
            best_loss = loss_value
            best_V = V.detach().clone()
            best_W = W.detach().clone()
            wait = 0
        else:
            wait += 1

        if step % 1000 == 0 or step == max_iter - 1:
            
            P = make_soft_matching(
                V,
                W,
                beta=beta,
                kind=kind,
                favor_mapper=favor_mapper,
            ).detach()
            P_np = P.cpu().numpy()
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(P_np, maximize=True)

            cnt = np.sum(row_ind == col_ind)

            # for acm-dblp
            acc_hungarian = cnt / n

            print(
                f"P_min={P_np.min():.6e} "
                f"P_max={P_np.max():.6e} "
                f"row_max_mean={P_np.max(axis=1).mean():.6e} "
                f"uniform={1.0 / n:.6e}"
            )
            
            print(
                f"step={step} "
                f"loss={float(loss.detach()):.6f} "
                f"structure={float(structure_term.detach()):.6f} "
                f"feature={float(feature_term.detach()):.6f} "
                f"penalty={float(constraint_term.detach()):.6f} "
                f"accuracy={acc_hungarian:.4f}"
            )

        if wait >= patience:
            print(f"Early stopping at step={step}, best_loss={best_loss:.6f}, accuracy={acc_hungarian:.4f}")
            break
    
    time_taken = time.time() - start_time
    print(f"Time taken for training with Adam: {time_taken:.2f} seconds")

    P_final = make_soft_matching(
        best_V,
        best_W,
        beta=beta,
        kind=kind,
        favor_mapper=favor_mapper,
    ).detach()
    return P_final, best_V, best_W, history



def nx_to_torch_sparse(G, n):
    # 1. 转成 scipy 稀疏矩阵（COO）
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format='coo')

    # 2. 转成 PyTorch sparse
    indices = torch.tensor([A.row, A.col], dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)

    A_torch = torch.sparse_coo_tensor(indices, values, size=(n, n))
    return A_torch.coalesce()   # 很重要


if __name__ == "__main__":
    
    # ## hyperparameters
    
    use_GPU = True
    learning_rate = 1e-3
    max_iter = 30000
    m_list = [500]
    beta_list = [10]
    row_penalty_list = [10]
    col_penalty_list = [200]
    mu = 0.5
    kind = "relu_squared"
    favor_features = 200



    Gq, Gt, n = read_file()


    for m in m_list:
        for beta in beta_list:
            for row_penalty in row_penalty_list:
                for col_penalty in col_penalty_list:
                    print(f"embed_dim={m} beta={beta} row_penalty={row_penalty} col_penalty={col_penalty}")
                    
                   

                    P_final, V_final, W_final, history = train_with_adam(
                        Gq,
                        Gt,
                        embed_dim=m,
                        beta=beta,
                        mu=mu,
                        row_penalty=row_penalty,
                        col_penalty=col_penalty,
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                        use_GPU=use_GPU,
                        kind=kind,
                        favor_features=favor_features)
                    
                    P_final_np = P_final.cpu().numpy()

                    row_sums = np.sum(P_final_np, axis=1)
                    col_sums = np.sum(P_final_np, axis=0)

                    rows_close_to_1 = np.allclose(row_sums, 1.0, atol=1e-2)
                    cols_close_to_1 = np.allclose(col_sums, 1.0, atol=1e-2)

                    row_diff = row_sums - 1.0
                    col_diff = col_sums - 1.0

                    row_max_abs_diff = np.max(np.abs(row_diff))
                    row_mean_abs_diff = np.mean(np.abs(row_diff))
                    col_max_abs_diff = np.max(np.abs(col_diff))
                    col_mean_abs_diff = np.mean(np.abs(col_diff))
                    
                    cnt = 0
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(P_final_np, maximize=True)



                    cnt = np.sum(row_ind == col_ind)
                    acc_hungarian = cnt / n

                    print("acc_hungarian:", acc_hungarian)

                    matched_cols = set()
                    match = -np.ones(n, dtype=int)

                    for i in range(n):
                        row = P_final_np[i]
                        candidates = np.argsort(-row)  # 从大到小排序
                        for j in candidates:
                            if j not in matched_cols:
                                match[i] = j
                                matched_cols.add(j)
                                break
                    acc_greedy = np.sum(match == np.arange(n)) / n
                    print("acc_greedy:", acc_greedy)
