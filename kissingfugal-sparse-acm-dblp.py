import networkx as nx
import torch
import numpy as np
from helpers.pred import feature_extraction
import scipy


path = "/home/cheng/Fugal/data/real_noise/ACM-DBLP/pos_pairs.npy"
data = np.load(path)
ground_truth = {pair[0]: pair[1] for pair in data}


def read_file():
    query_path = "/home/cheng/fly/data/real_noise/ACM-DBLP/ACM.txt"
    target_path = "/home/cheng/fly/data/real_noise/ACM-DBLP/DBLP.txt"

    n = 9916
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


def build_distance_candidate_pool(F1, F2, k, device, dtype=torch.float32, block_size=512):
    """For each row in F1, keep the k nearest rows in F2 without storing n x n."""
    F1_t = torch.as_tensor(F1, device=device, dtype=dtype)
    F2_t = torch.as_tensor(F2, device=device, dtype=dtype)
    n = F1_t.shape[0]
    k = min(k, F2_t.shape[0])
    chunks = []

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        dist = torch.cdist(F1_t[start:end], F2_t, p=2)
        chunks.append(torch.topk(dist, k=k, largest=False, dim=1).indices)

    return torch.cat(chunks, dim=0)


def sample_candidate_columns(candidate_pool, random_r, n):
    """Keep every top-k candidate, then add r random columns outside that top-k set."""
    num_rows, k = candidate_pool.shape
    random_r = min(random_r, max(n - k, 0))
    if random_r <= 0:
        return candidate_pool

    random_cols = torch.empty((num_rows, random_r), device=candidate_pool.device, dtype=candidate_pool.dtype)
    filled = torch.zeros(num_rows, device=candidate_pool.device, dtype=torch.long)
    row_idx = torch.arange(num_rows, device=candidate_pool.device)

    while torch.any(filled < random_r):
        active = filled < random_r
        proposals = torch.randint(0, n, (int(active.sum()),), device=candidate_pool.device)
        active_rows = row_idx[active]
        in_topk = (candidate_pool[active_rows] == proposals[:, None]).any(dim=1)
        accepted_rows = active_rows[~in_topk]
        accepted_cols = proposals[~in_topk]

        if accepted_cols.numel() == 0:
            continue

        slots = filled[accepted_rows]
        random_cols[accepted_rows, slots] = accepted_cols
        filled[accepted_rows] += 1

    return torch.cat([candidate_pool, random_cols], dim=1)


def make_sampled_soft_matching(V, W, candidate_cols, beta=20.0, block_size=256):
    """Build sparse row-wise softmax values over sampled candidate columns."""
    n, r = candidate_cols.shape
    Vn = V / torch.linalg.norm(V, dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / torch.linalg.norm(W, dim=1, keepdim=True).clamp_min(1e-12)

    score_chunks = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        scores_full = Vn[start:end] @ Wn.T
        score_chunks.append(scores_full.gather(1, candidate_cols[start:end]))

    scores = torch.cat(score_chunks, dim=0)
    probs = torch.softmax(beta * scores, dim=1)
    row_idx = torch.arange(n, device=V.device).repeat_interleave(r)
    col_idx = candidate_cols.reshape(-1)
    return row_idx, col_idx, probs, probs.reshape(-1)


def greedy_match_from_candidates(candidate_cols, probs):
    n = candidate_cols.shape[0]
    order = torch.argsort(probs, dim=1, descending=True).detach().cpu().numpy()
    candidates = candidate_cols.detach().cpu().numpy()
    matched_cols = set()
    match = -np.ones(n, dtype=int)

    for i in range(n):
        for pos in order[i]:
            col = int(candidates[i, pos])
            if col not in matched_cols:
                match[i] = col
                matched_cols.add(col)
                break

    return match


def ground_truth_accuracy(match):
    cnt = 0
    for row, col in enumerate(match):
        gt = ground_truth.get(int(row))
        if gt is not None and gt == int(col):
            cnt += 1
    return cnt / data.shape[0], cnt


def hungarian_accuracy(P_np):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(P_np, maximize=True)
    cnt = 0
    for row, col in zip(row_ind, col_ind):
        gt = ground_truth.get(int(row))
        if gt is not None and gt == int(col):
            cnt += 1
    return cnt / data.shape[0], cnt


def candidate_ground_truth_recall(candidate_pool):
    """统计 ground truth target 是否出现在每个 source row 的候选集合中。"""
    hits = 0
    total = 0
    for row, col in ground_truth.items():
        if row >= candidate_pool.shape[0]:
            continue
        total += 1
        if torch.any(candidate_pool[int(row)] == int(col)):
            hits += 1

    if total == 0:
        return 0.0
    return hits / total


def build_dense_edge_mask(B, n):
    """把 B 图的边存成 bool 矩阵，后面可以 O(1) 查询候选边是否存在。"""
    B = B.coalesce()
    edge_mask = torch.zeros((n, n), device=B.device, dtype=torch.bool)
    edge_mask[B.indices()[0], B.indices()[1]] = True
    return edge_mask


class SampledStructureTerm(torch.autograd.Function):
    """低内存计算结构项，避免显式构造 P、AP、PB 这些稀疏中间矩阵。"""

    @staticmethod
    def forward(ctx, probs, candidate_cols, A_edge_index, B_edge_mask, chunk_size):
        total = probs.new_zeros(())

        # 按 A 的边分块计算，临时张量大小约为 chunk_size x r x r。
        # 这些临时张量不会被 PyTorch autograd 自动保存，反向传播由下面 backward 手写。
        for start in range(0, A_edge_index.shape[1], chunk_size):
            end = min(start + chunk_size, A_edge_index.shape[1])
            src = A_edge_index[0, start:end]
            dst = A_edge_index[1, start:end]

            src_cols = candidate_cols[src]
            dst_cols = candidate_cols[dst]
            src_probs = probs[src]
            dst_probs = probs[dst]

            # edge_exists[e, a, b] 表示：
            # 对 A 中第 e 条边 (src, dst)，B 中是否存在候选边
            # (candidate_cols[src, a], candidate_cols[dst, b])。
            edge_exists = B_edge_mask[src_cols[:, :, None], dst_cols[:, None, :]]
            pair_probs = src_probs[:, :, None] * dst_probs[:, None, :]
            total = total + (pair_probs * edge_exists.to(probs.dtype)).sum()

        ctx.save_for_backward(probs, candidate_cols, A_edge_index, B_edge_mask)
        ctx.chunk_size = chunk_size
        return -total

    @staticmethod
    def backward(ctx, grad_output):
        probs, candidate_cols, A_edge_index, B_edge_mask = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        grad_probs = torch.zeros_like(probs)

        # 手写结构项对 probs 的梯度，避免 autograd 保存每个分块的 r x r 中间结果。
        for start in range(0, A_edge_index.shape[1], chunk_size):
            end = min(start + chunk_size, A_edge_index.shape[1])
            src = A_edge_index[0, start:end]
            dst = A_edge_index[1, start:end]

            src_cols = candidate_cols[src]
            dst_cols = candidate_cols[dst]
            src_probs = probs[src]
            dst_probs = probs[dst]

            edge_exists = B_edge_mask[src_cols[:, :, None], dst_cols[:, None, :]].to(probs.dtype)

            # 对 P[src, a] 的梯度：累加所有与它在 B 中成边的 P[dst, b]。
            grad_src = torch.sum(edge_exists * dst_probs[:, None, :], dim=2)
            # 对 P[dst, b] 的梯度：累加所有与它在 B 中成边的 P[src, a]。
            grad_dst = torch.sum(edge_exists * src_probs[:, :, None], dim=1)

            # 因为结构项是负号：-sum P[src,a] * P[dst,b]。
            grad_probs.index_add_(0, src, -grad_src)
            grad_probs.index_add_(0, dst, -grad_dst)

        return grad_output * grad_probs, None, None, None, None


def sparse_candidate_graph_matching_loss(
    row_idx,
    col_idx,
    prob_values,
    probs,
    candidate_cols,
    A_edge_index,
    B_edge_mask,
    F1,
    F2,
    n,
    mu=1.0,
    col_penalty=1.0,
    structure_chunk_size=32,
):
    # 结构项原来通过 P -> AP -> PB 计算，会产生很多 COO 稀疏中间矩阵。
    # 这里直接沿 A 的边和 B 的候选边查询计算，避免 materialize P/AP/PB。
    structure_term = SampledStructureTerm.apply(
        probs,
        candidate_cols,
        A_edge_index,
        B_edge_mask,
        structure_chunk_size,
    )

    # 特征项只在候选非零位置上计算距离，不构造完整 n x n 距离矩阵。
    distances = torch.linalg.norm(F1[row_idx] - F2[col_idx], dim=1)
    feature_term = mu * torch.sum(prob_values * distances)

    # 行约束由 softmax 天然保证为 1，所以这里不再计算 row penalty。
    # 列约束仍然需要 scatter_add，因为同一个 target 列可能被多行候选选中。
    col_sum = torch.zeros(n, device=prob_values.device, dtype=prob_values.dtype)
    col_sum.scatter_add_(0, col_idx, prob_values)
    constraint_term = col_penalty * torch.sum((col_sum - 1.0) ** 2)

    loss = structure_term + feature_term + constraint_term
    return loss, structure_term, feature_term, constraint_term


def train_with_adam(
    Gq: nx.Graph,
    Gt: nx.Graph,
    embed_dim: int = 30,
    beta: float = 10.0,
    mu: float = 1.0,
    col_penalty: float = 10.0,
    learning_rate: float = 1e-2,
    max_iter: int = 1000,
    use_GPU: bool = True,
    candidate_k: int = 200,
    sample_r: int = 50,
    candidate_block_size: int = 512,
    structure_chunk_size: int = 32,
):
    n = Gq.number_of_nodes()
    if Gt.number_of_nodes() != n:
        raise ValueError("This prototype assumes Gq and Gt have the same number of nodes.")

    device = torch.device("cuda" if use_GPU and torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    candidate_pool = build_distance_candidate_pool(
        F1, F2, candidate_k, device=device, dtype=dtype, block_size=candidate_block_size
    )
    gt_recall = candidate_ground_truth_recall(candidate_pool)
    print(
        f"candidate_pool: top_k={candidate_pool.shape[1]} "
        f"random_r={min(sample_r, max(n - candidate_pool.shape[1], 0))} "
        f"gt_recall={gt_recall:.4f}"
    )

    A = nx_to_torch_sparse(Gq, n)
    B = nx_to_torch_sparse(Gt, n)

    A = A.to(device, dtype=dtype)
    B = B.to(device, dtype=dtype)
    A_edge_index = A.indices()
    B_edge_mask = build_dense_edge_mask(B, n)
    F1 = torch.tensor(F1, device=device, dtype=dtype)
    F2 = torch.tensor(F2, device=device, dtype=dtype)
    F1 = F1.to(device)
    F2 = F2.to(device)
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

    for step in range(max_iter):
        candidate_cols = sample_candidate_columns(candidate_pool, sample_r, n)
        row_idx, col_idx, probs, prob_values = make_sampled_soft_matching(
            V, W, candidate_cols, beta=beta
        )
        loss, structure_term, feature_term, constraint_term = sparse_candidate_graph_matching_loss(
            row_idx,
            col_idx,
            prob_values,
            probs,
            candidate_cols,
            A_edge_index,
            B_edge_mask,
            F1,
            F2,
            n,
            mu=mu,
            col_penalty=col_penalty,
            structure_chunk_size=structure_chunk_size,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(float(loss.detach()))

        loss_value = float(loss.detach())

        if loss_value < best_loss - min_delta:
            best_loss = loss_value
            best_V = V.detach().clone()
            best_W = W.detach().clone()
            wait = 0
        else:
            wait += 1

        if step % 1000 == 0 or step == max_iter - 1:
            match = greedy_match_from_candidates(candidate_cols, probs)
            acc_greedy, cnt = ground_truth_accuracy(match)
            print(
                f"step={step} "
                f"loss={float(loss.detach()):.6f} "
                f"structure={float(structure_term.detach()):.6f} "
                f"feature={float(feature_term.detach()):.6f} "
                f"penalty={float(constraint_term.detach()):.6f} "
                f"accuracy={acc_greedy:.4f}"
            )

        if wait >= patience:
            match = greedy_match_from_candidates(candidate_cols, probs)
            acc_greedy, cnt = ground_truth_accuracy(match)
            print(f"Early stopping at step={step}, best_loss={best_loss:.6f}, accuracy={acc_greedy:.4f}")
            break

    final_candidate_cols = sample_candidate_columns(candidate_pool, sample_r, n)
    _, _, final_probs, _ = make_sampled_soft_matching(
        best_V, best_W, final_candidate_cols, beta=beta
    )
    P_final = (final_candidate_cols.detach(), final_probs.detach())
    return P_final, best_V, best_W, history


def nx_to_torch_sparse(G, n):
    # 1. 转成 scipy 稀疏矩阵（COO）
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="coo")

    # 2. 转成 PyTorch sparse
    indices = torch.tensor(np.vstack((A.row, A.col)), dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)

    with torch.sparse.check_sparse_tensor_invariants(False):
        A_torch = torch.sparse_coo_tensor(indices, values, size=(n, n))
    return A_torch.coalesce()   # 很重要


if __name__ == "__main__":
    use_GPU = True
    learning_rate = 1e-2
    max_iter = 10000
    m_list = [1000]
    beta_list = [10]
    col_penalty_list = [200]
    candidate_k = 2000
    sample_r = 0
    structure_chunk_size = 32
    mu = 0.1  # weight for the feature term in the loss function

    Gq, Gt, n = read_file()

    for m in m_list:
        for beta in beta_list:
            for col_penalty in col_penalty_list:
                print(f"embed_dim={m} beta={beta} col_penalty={col_penalty}")

                P_final, V_final, W_final, history = train_with_adam(
                    Gq,
                    Gt,
                    embed_dim=m,
                    beta=beta,
                    mu=mu,
                    col_penalty=col_penalty,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    use_GPU=use_GPU,
                    candidate_k=candidate_k,
                    sample_r=sample_r,
                    structure_chunk_size=structure_chunk_size,
                )

                candidate_cols, final_probs = P_final
                candidate_cols_np = candidate_cols.cpu().numpy()
                final_probs_np = final_probs.cpu().numpy()

                if candidate_cols_np.shape[1] == n:
                    P_final_np = np.zeros((n, n), dtype=final_probs_np.dtype)
                    P_final_np[
                        np.repeat(np.arange(n), candidate_cols_np.shape[1]),
                        candidate_cols_np.reshape(-1),
                    ] = final_probs_np.reshape(-1)
                    acc_hungarian, cnt = hungarian_accuracy(P_final_np)
                    print("acc_hungarian:", acc_hungarian)
                else:
                    match = greedy_match_from_candidates(candidate_cols, final_probs)
                    acc_greedy, cnt = ground_truth_accuracy(match)
                    print("acc_greedy:", acc_greedy)
