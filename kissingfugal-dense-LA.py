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


def kernel_feature(X, kind="elu"):
    if kind == "elu":
        return F.elu(X) + 1.0
    elif kind == "softplus":
        return F.softplus(X) + 1e-6
    elif kind == "exp":
        return torch.exp(X.clamp(max=10))
    else:
        raise ValueError(kind)


def linear_matching_factors(
    V: torch.Tensor,
    W: torch.Tensor,
    beta: float = 20.0,
    kind: str = "elu"
):
    """Return factors for P ~= R @ K.T without materializing the n x n matrix."""
    Vn = V / torch.linalg.norm(V, dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / torch.linalg.norm(W, dim=1, keepdim=True).clamp_min(1e-12)
    # Vn = V
    # Wn = W

    scale = math.sqrt(beta)
    Q = kernel_feature(scale * Vn, kind=kind)
    K = kernel_feature(scale * Wn, kind=kind)
    denom = (Q @ K.sum(dim=0)).clamp_min(1e-12)
    R = Q / denom[:, None]
    return R, K


def make_soft_matching(
    V: torch.Tensor,
    W: torch.Tensor,
    beta: float = 20.0,
) -> torch.Tensor:
    """Build P with a linear-attention kernel form.

    P_ij ~= phi(sqrt(beta) V_i)^T phi(sqrt(beta) W_j)
            / sum_l phi(sqrt(beta) V_i)^T phi(sqrt(beta) W_l)
    """
    R, K = linear_matching_factors(V, W, beta=beta)
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
    kind: str = "elu"
):
    R, K = linear_matching_factors(V, W, beta=beta, kind="elu")

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
            kind="softplus"
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
            
            P = make_soft_matching(V, W, beta=beta).detach()
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

    P_final = make_soft_matching(best_V, best_W, beta=beta).detach()
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
    learning_rate = 1e-2
    max_iter = 30000
    m_list = [100]
    beta_list = [10]
    row_penalty_list = [10]
    col_penalty_list = [200]
    mu = 0.5  




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
                        use_GPU=use_GPU)
                    
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
