import networkx as nx
import torch
import numpy as np
from helpers.pred import feature_extraction, convertToPermHungarian
import scipy
import csv
import itertools

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


def build_D1_D2(Gq: nx.Graph, Gt: nx.Graph, dtype=torch.float32):
    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    return build_D1_D2_from_features(F1, F2)


def build_D1_D2_from_features(F1, F2):

    """
    F1: (n, d) feature matrix of graph 1
    F2: (n, d) feature matrix of graph 2

    Returns:
        D1: (n, d+2)
        D2: (n, d+2)
    """

    # X^T and Y^T
    XT = F1.T          # (d, n)
    YT = F2.T          # (d, n)

    n, d = F1.shape

    # p = ||x_i||^2
    p = np.sum(F1**2, axis=1)   # (n,)

    # q = ||y_j||^2
    q = np.sum(F2**2, axis=1)   # (n,)

    # reshape for concatenation
    p = p.reshape(-1, 1)        # (n,1)
    q = q.reshape(-1, 1)        # (n,1)
    ones = np.ones((n, 1))      # (n,1)

    # D1 = [p, 1, -2X^T]
    D1 = np.hstack([
        p,
        ones,
        -2 * XT.T   # (n,d)
    ])

    # D2 = [1, q, Y^T]
    D2 = np.hstack([
        ones,
        q,
        YT.T        # (n,d)
    ])

    return D1, D2


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


def make_soft_matching(V: torch.Tensor, W: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """Build a soft matching matrix P from row embeddings V and W."""
    Vn = V / torch.linalg.norm(V, dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / torch.linalg.norm(W, dim=1, keepdim=True).clamp_min(1e-12)
    scores = Vn @ Wn.T
    return torch.softmax(beta * scores, dim=1)


def make_sampled_soft_matching(V, W, candidate_cols, beta=20.0):
    """Build sparse row-wise softmax values over sampled candidate columns."""
    n, r = candidate_cols.shape
    Vn = V / torch.linalg.norm(V, dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / torch.linalg.norm(W, dim=1, keepdim=True).clamp_min(1e-12)
    selected_W = Wn[candidate_cols]
    scores = torch.sum(Vn[:, None, :] * selected_W, dim=2)
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


def sparse_candidate_graph_matching_loss(
    row_idx,
    col_idx,
    prob_values,
    probs,
    A,
    B,
    F1,
    F2,
    n,
    mu=1.0,
    row_penalty=1.0,
    col_penalty=1.0,
):
    indices = torch.stack([row_idx, col_idx])
    P = torch.sparse_coo_tensor(indices, prob_values, size=(n, n)).coalesce()

    AP = torch.sparse.mm(A, P)
    PB = torch.sparse.mm(P, B.transpose(0, 1))

    if AP.is_sparse:
        structure_term = -torch.sum((AP * PB).coalesce().values())
    else:
        structure_term = -torch.sum(AP * PB)

    distances = torch.linalg.norm(F1[row_idx] - F2[col_idx], dim=1)
    feature_term = mu * torch.sum(prob_values * distances)

    col_sum = torch.zeros(n, device=prob_values.device, dtype=prob_values.dtype)
    col_sum.scatter_add_(0, col_idx, prob_values)
    constraint_term = (
        row_penalty * torch.sum((torch.sum(probs, dim=1) - 1.0) ** 2) +
        col_penalty * torch.sum((col_sum - 1.0) ** 2)
    )

    loss = structure_term + feature_term + constraint_term
    return loss, structure_term, feature_term, constraint_term



def sparse_softmax_graph_matching_loss(
    P,                 # (n, n)
    A, B,              # (n, n) sparse or dense
    D1, D2,            # D = D1 @ D2.T
    mu=1.0,
    row_penalty=1.0,
    col_penalty=1.0,
):
    n = P.shape[0]

    # =========================
    # 1. STRUCTURE TERM
    # -tr(A P B^T P^T)
    # =========================
    if A.is_sparse:
        AP = torch.sparse.mm(A, P)
    else:
        AP = A @ P

    if B.is_sparse:
        PB = torch.sparse.mm(B.transpose(0, 1), P.T).T
    else:
        PB = P @ B

    structure_term = -torch.sum(AP * PB)


    # =========================
    # 2. FEATURE TERM
    # mu * tr(P^T D)
    # =========================
    feature_term = mu * torch.sum((P.T @ D1) * D2)


    # =========================
    # 3. CONSTRAINT TERM
    # doubly stochastic penalty
    # =========================
    constraint_term = (
        row_penalty * torch.sum((torch.sum(P, dim=1) - 1.0) ** 2) +
        col_penalty * torch.sum((torch.sum(P, dim=0) - 1.0) ** 2)
    )


    # =========================
    # TOTAL LOSS
    # =========================
    loss = structure_term + feature_term + constraint_term

    return loss, structure_term, feature_term, constraint_term


def fugal_loss(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    P: torch.Tensor,
    mu: float,
    row_penalty: float,
    col_penalty: float,
) -> torch.Tensor:
    structure_term = -torch.trace(A @ P @ B.T @ P.T)
    feature_term = mu * torch.trace(P.T @ D)
    constraint_term = row_penalty * torch.sum((torch.sum(P, dim=1) - 1.0) ** 2) + col_penalty * torch.sum((torch.sum(P, dim=0) - 1.0) ** 2)
    return structure_term + feature_term + constraint_term

def fugal_loss_terms(
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    P: torch.Tensor,
    mu: float,
    row_penalty: float,
    col_penalty: float,
):
    structure_term = -torch.trace(A @ P @ B.T @ P.T)
    feature_term = mu * torch.trace(P.T @ D)
    constraint_term = row_penalty * torch.sum((torch.sum(P, dim=1) - 1.0) ** 2) + col_penalty * torch.sum((torch.sum(P, dim=0) - 1.0) ** 2)
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
    max_iter: int = 1000,
    use_GPU: bool = True,
    candidate_k: int = 200,
    sample_r: int = 50,
    candidate_block_size: int = 512,
):
    n = Gq.number_of_nodes()
    if Gt.number_of_nodes() != n:
        raise ValueError("This prototype assumes Gq and Gt have the same number of nodes.")
    
    device = torch.device("cuda" if use_GPU and torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    D1, D2 = build_D1_D2_from_features(F1, F2)
    candidate_pool = build_distance_candidate_pool(
        F1, F2, candidate_k, device=device, dtype=dtype, block_size=candidate_block_size
    )
    identity_recall = torch.mean(
        (candidate_pool == torch.arange(n, device=device)[:, None]).any(dim=1).float()
    )
    print(
        f"candidate_pool: top_k={candidate_pool.shape[1]} "
        f"random_r={min(sample_r, max(n - candidate_pool.shape[1], 0))} "
        f"identity_recall={float(identity_recall):.4f}"
    )

    A = nx_to_torch_sparse(Gq, n)
    B = nx_to_torch_sparse(Gt, n)

    A = A.to(device,dtype=dtype)
    B = B.to(device,dtype=dtype)
    F1 = torch.tensor(F1, device=device, dtype=dtype)
    F2 = torch.tensor(F2, device=device, dtype=dtype)
    D1 = torch.tensor(D1, device=device, dtype=dtype)
    D2 = torch.tensor(D2, device=device, dtype=dtype)
    F1 = F1.to(device)
    F2 = F2.to(device)
    D1 = D1.to(device)
    D2 = D2.to(device)
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
            A,
            B,
            F1,
            F2,
            n,
            mu=mu,
            row_penalty=row_penalty,
            col_penalty=col_penalty,
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
            cnt = np.sum(match == np.arange(n))
            print(
                f"step={step} "
                f"loss={float(loss.detach()):.6f} "
                f"structure={float(structure_term.detach()):.6f} "
                f"feature={float(feature_term.detach()):.6f} "
                f"penalty={float(constraint_term.detach()):.6f} "
                f"accuracy={cnt / n:.4f}"
            )

        if wait >= patience:
            match = greedy_match_from_candidates(candidate_cols, probs)
            cnt = np.sum(match == np.arange(n))
            print(f"Early stopping at step={step}, best_loss={best_loss:.6f}, accuracy={cnt / n:.4f}")
            break

    final_candidate_cols = sample_candidate_columns(candidate_pool, sample_r, n)
    _, _, final_probs, _ = make_sampled_soft_matching(
        best_V, best_W, final_candidate_cols, beta=beta
    )
    P_final = (final_candidate_cols.detach(), final_probs.detach())
    return P_final, best_V, best_W, history

def nx_to_torch_sparse(G, n):
    # 1. 转成 scipy 稀疏矩阵（COO）
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format='coo')

    # 2. 转成 PyTorch sparse
    indices = torch.tensor(np.vstack((A.row, A.col)), dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)

    with torch.sparse.check_sparse_tensor_invariants(False):
        A_torch = torch.sparse_coo_tensor(indices, values, size=(n, n))
    return A_torch.coalesce()   # 很重要

if __name__ == "__main__":
    
    # ## hyperparameters

    # m = 200

    # beta = 20.0 # controls the sharpness of the soft matching
    # row_penalty = 50.0 # penalty weight for doubly stochastic constraint
    # col_penalty = 100.0 # penalty weight for doubly stochastic constraint
    # max_iter = 50000 # iterations for training
    # ## hyperparameters
    
    use_GPU = True
    learning_rate = 1e-2
    max_iter = 30000
    m_list = [100]
    beta_list = [10]
    row_penalty_list = [10]
    col_penalty_list = [200]
    candidate_k = 100
    sample_r = 50
    mu = 0.5  # weight for the feature term in the loss function


    # output_file = "grid_search_results.csv"

    # with open(output_file, "w", newline="") as f:
    #     writer = csv.DictWriter(
    #         f,
    #         fieldnames=[
    #             "embed_dim",
    #             "beta",
    #             "row_penalty",
    #             "col_penalty",
    #             "rows_close_to_1",
    #             "cols_close_to_1",
    #             "row_max_abs_diff",
    #             "row_mean_abs_diff",
    #             "col_max_abs_diff",
    #             "col_mean_abs_diff",
    #             "num_correct_matches",
    #             "accuracy",
    #             "final_loss",
    #         ],
    #     )
    #     writer.writeheader()


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
                        candidate_k=candidate_k,
                        sample_r=sample_r)
                    
                    candidate_cols, final_probs = P_final
                    candidate_cols_np = candidate_cols.cpu().numpy()
                    final_probs_np = final_probs.cpu().numpy()

                    row_sums = np.sum(final_probs_np, axis=1)
                    col_sums = np.zeros(n, dtype=final_probs_np.dtype)
                    np.add.at(col_sums, candidate_cols_np.reshape(-1), final_probs_np.reshape(-1))

                    rows_close_to_1 = np.allclose(row_sums, 1.0, atol=1e-2)
                    cols_close_to_1 = np.allclose(col_sums, 1.0, atol=1e-2)

                    row_diff = row_sums - 1.0
                    col_diff = col_sums - 1.0

                    row_max_abs_diff = np.max(np.abs(row_diff))
                    row_mean_abs_diff = np.mean(np.abs(row_diff))
                    col_max_abs_diff = np.max(np.abs(col_diff))
                    col_mean_abs_diff = np.mean(np.abs(col_diff))
                    
                    if candidate_cols_np.shape[1] == n:
                        P_final_np = np.zeros((n, n), dtype=final_probs_np.dtype)
                        P_final_np[
                            np.repeat(np.arange(n), candidate_cols_np.shape[1]),
                            candidate_cols_np.reshape(-1),
                        ] = final_probs_np.reshape(-1)
                        row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                            P_final_np, maximize=True
                        )
                        acc_hungarian = np.sum(row_ind == col_ind) / n
                        print("acc_hungarian:", acc_hungarian)
                    else:
                        match = greedy_match_from_candidates(candidate_cols, final_probs)
                        acc_greedy = np.sum(match == np.arange(n)) / n
                        print("acc_greedy:", acc_greedy)


    #                     writer.writerow({
    #                         "embed_dim": m,
    #                         "beta": beta,
    #                         "row_penalty": row_penalty,
    #                         "col_penalty": col_penalty,
    #                         "rows_close_to_1": rows_close_to_1,
    #                         "cols_close_to_1": cols_close_to_1,
    #                         "row_max_abs_diff": row_max_abs_diff,
    #                         "row_mean_abs_diff": row_mean_abs_diff,
    #                         "col_max_abs_diff": col_max_abs_diff,
    #                         "col_mean_abs_diff": col_mean_abs_diff,
    #                         "num_correct_matches": int(cnt),
    #                         "accuracy": float(acc),
    #                         "final_loss": float(history[-1]),
    #                     })
    #                     f.flush()

    # print(f"Saved results to {output_file}")
