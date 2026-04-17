import networkx as nx
import torch
import numpy as np
from helpers.pred import feature_extraction, convertToPermHungarian
import scipy
import csv
import itertools

def read_file():
    query_path = "/home/cheng/fly/data/real_noise/contacts-prox-high-school-2013/contacts-prox-high-school-2013_100.txt"
    target_path = "/home/cheng/fly/data/real_noise/contacts-prox-high-school-2013/contacts-prox-high-school-2013_80.txt"

    n = 327
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


def make_soft_matching(V: torch.Tensor, W: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    """Build a soft matching matrix P from row embeddings V and W."""
    Vn = V / torch.linalg.norm(V, dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / torch.linalg.norm(W, dim=1, keepdim=True).clamp_min(1e-12)
    scores = Vn @ Wn.T
    return torch.softmax(beta * scores, dim=1)

# def doubly_stochastic_penalty(P: torch.Tensor) -> torch.Tensor:
#     row_penalty = torch.sum((torch.sum(P, dim=1) - 1.0) ** 2)
#     col_penalty = torch.sum((torch.sum(P, dim=0) - 1.0) ** 2)
#     return row_penalty + col_penalty

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
):
    n = Gq.number_of_nodes()
    if Gt.number_of_nodes() != n:
        raise ValueError("This prototype assumes Gq and Gt have the same number of nodes.")
    
    device = torch.device("cuda" if use_GPU else "cpu")
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
    patience = 50
    min_delta = 1e-4

    for step in range(max_iter):
        P = make_soft_matching(V, W, beta=beta)
        loss, structure_term, feature_term, constraint_term = fugal_loss_terms(
            A, B, D, P, mu=mu, row_penalty=row_penalty, col_penalty=col_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(float(loss.detach()))

        P_np = P.detach().cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(P_np, maximize=True)

        cnt = np.sum(row_ind == col_ind)

        loss_value = float(loss.detach())

        if loss_value < best_loss - min_delta:
            best_loss = loss_value
            best_V = V.detach().clone()
            best_W = W.detach().clone()
            wait = 0
        else:
            wait += 1

        if step % 1000 == 0 or step == max_iter - 1:
            print(
                f"step={step} "
                f"loss={float(loss.detach()):.6f} "
                f"structure={float(structure_term.detach()):.6f} "
                f"feature={float(feature_term.detach()):.6f} "
                f"penalty={float(constraint_term.detach()):.6f} "
                f"accuracy={cnt / n:.4f}"
            )

        if wait >= patience:
            print(f"Early stopping at step={step}, best_loss={best_loss:.6f}, accuracy={cnt / n:.4f}")
            break

    P_final = make_soft_matching(best_V, best_W, beta=beta).detach()
    return P_final, best_V, best_W, history


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
    m_list = [10,20,30]
    beta_list = [10]
    row_penalty_list = [10]
    col_penalty_list = [200]
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
