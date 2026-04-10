import networkx as nx
import numpy as np
import scipy
import torch

from helpers.pred import feature_extraction
from helpers import sinkhorn


def eucledian_dist(F1, F2):
    if not torch.is_tensor(F1):
        F1 = torch.tensor(F1, dtype=torch.float64)
    if not torch.is_tensor(F2):
        F2 = torch.tensor(F2, dtype=torch.float64)
    return torch.cdist(F1, F2, p=2)


def sparse_to_dense(idx, P_sparse, n):
    P_dense = torch.zeros((n, n), dtype=P_sparse.dtype, device=P_sparse.device)
    P_dense.scatter_(1, idx, P_sparse)
    return P_dense


def dense_to_sparse(idx, M_dense):
    return torch.gather(M_dense, 1, idx)


def chunked_topk_support(F1, F2, k, chunk_size=256):
    """
    Compute row-wise top-k supports without materializing the full n x n distance matrix.
    """
    n = F1.shape[0]
    idx_chunks = []
    dist_chunks = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        D_chunk = torch.cdist(F1[start:end], F2, p=2)
        vals, inds = torch.topk(D_chunk, k=k, dim=1, largest=False)
        idx_chunks.append(inds.cpu())
        dist_chunks.append(vals.cpu())

    idx = torch.cat(idx_chunks, dim=0).numpy().astype(np.int64, copy=False)
    Dk = torch.cat(dist_chunks, dim=0).numpy().astype(np.float64, copy=False)
    return idx, Dk


def build_sparse_adjacency(G):
    adj = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
    return scipy.sparse.csr_matrix(adj)


def compute_sparse_gradient_n_k(A, AT, B_csr, B_csc, P, idx, Dk, mu, lam):
    """
    Gradient on the n x k support only.
    """
    n, k = P.shape
    G = np.empty((n, k), dtype=np.float64)
    unique_js = np.unique(idx)

    for j in unique_js:
        b_col = B_csc.getcol(int(j)).toarray().ravel()
        b_row = B_csr.getrow(int(j)).toarray().ravel()

        x = (P * b_col[idx]).sum(axis=1)
        y = (P * b_row[idx]).sum(axis=1)

        term1 = AT.dot(x)
        term2 = A.dot(y)

        pos = idx == j
        row_ids = np.nonzero(pos)[0]

        G[pos] = (
            -term1[row_ids]
            -term2[row_ids]
            + mu * Dk[pos]
            + lam * (1.0 - 2.0 * P[pos])
        )

    return G


def sparse_transport_update(idx, G, reg, beta=0.35, maxIter=50, stopThr=1e-5):
    """
    Relaxed sparse transport update.

    This keeps the update in O(nk) memory:
    - row normalization is enforced exactly
    - column competition is enforced softly through a power-law rebalance
    """
    eps = 1e-12
    G_shift = G - G.min()
    K = np.exp(-G_shift / reg).clip(min=eps)
    q = K / np.clip(K.sum(axis=1, keepdims=True), eps, None)

    n = idx.shape[0]
    for _ in range(maxIter):
        prev = q
        col_sum = np.bincount(idx.reshape(-1), weights=q.reshape(-1), minlength=n)
        scale = np.power(np.clip(col_sum[idx], eps, None), -beta)
        q = K * scale
        q = q / np.clip(q.sum(axis=1, keepdims=True), eps, None)

        if np.max(np.abs(q - prev)) < stopThr:
            break

    return q


def sparse_hungarian(idx, P, n1, n2):
    """
    Sparse matching on the supported edges only.
    Falls back to greedy completion if a full sparse matching is infeasible.
    """
    n, k = idx.shape
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    scores = P.reshape(-1)

    max_score = float(scores.max()) if scores.size else 0.0
    costs = (max_score - scores) + 1e-12
    graph = scipy.sparse.coo_matrix((costs, (rows, cols)), shape=(n, n)).tocsr()

    try:
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(graph)
        ans = []
        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            if i < n1 and j < n2:
                ans.append((i, j))
        return ans
    except Exception:
        order = np.argsort(-scores, kind="stable")
        used_rows = np.zeros(n, dtype=bool)
        used_cols = np.zeros(n, dtype=bool)
        ans = []

        for pos in order:
            i = rows[pos]
            j = cols[pos]
            if used_rows[i] or used_cols[j]:
                continue
            used_rows[i] = True
            used_cols[j] = True
            if i < n1 and j < n2:
                ans.append((int(i), int(j)))

        remaining_rows = [i for i in range(n1) if not used_rows[i]]
        remaining_cols = [j for j in range(n2) if not used_cols[j]]
        for i, j in zip(remaining_rows, remaining_cols):
            ans.append((int(i), int(j)))
        return ans


def FindQuasiPerm_n_k_sparse(
    A,
    AT,
    B_csr,
    B_csc,
    Dk,
    idx,
    mu,
    niter,
    reg=1.0,
    transport_iter=50,
    transport_thr=1e-5,
    beta=0.35,
):
    """
    True sparse version: no n x n P, D, G, or q are materialized.
    """
    k = idx.shape[1]
    P = np.full((idx.shape[0], k), 1.0 / k, dtype=np.float64)

    for outer in range(niter):
        for it in range(1, 11):
            G = compute_sparse_gradient_n_k(
                A=A,
                AT=AT,
                B_csr=B_csr,
                B_csc=B_csc,
                P=P,
                idx=idx,
                Dk=Dk,
                mu=mu,
                lam=outer,
            )
            q = sparse_transport_update(
                idx=idx,
                G=G,
                reg=reg,
                beta=beta,
                maxIter=transport_iter,
                stopThr=transport_thr,
            )

            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
            P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

    return P


def FindQuasiPerm_n_k_hybrid(A, B, D, mu, niter, idx, reg=1.0, sinkhorn_iter=500, sinkhorn_thr=1e-5):
    """
    Hybrid baseline that stores P as n x k but computes q with dense semantics.
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device
    eps = 1e-12
    k = idx.shape[1]

    ones = torch.ones(n, dtype=dtype, device=device)
    mat_ones = torch.ones((n, n), dtype=dtype, device=device)
    P = torch.full((n, k), 1.0 / k, dtype=dtype, device=device)

    idx_t = torch.tensor(idx, dtype=torch.long, device=device)

    for outer in range(niter):
        for it in range(1, 11):
            P_dense = sparse_to_dense(idx_t, P, n)
            G_dense = (
                -torch.mm(torch.mm(A.T, P_dense), B)
                - torch.mm(torch.mm(A, P_dense), B.T)
                + mu * D
                + outer * (mat_ones - 2.0 * P_dense)
            )

            q_dense = sinkhorn.sinkhorn(
                ones,
                ones,
                G_dense,
                reg,
                maxIter=sinkhorn_iter,
                stopThr=sinkhorn_thr,
            )
            q = dense_to_sparse(idx_t, q_dense)
            q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)

            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
            P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)

    return P.detach().cpu().numpy()


def fly_n_k(
    Gq,
    Gt,
    k,
    mu=0.5,
    niter=15,
    reg=1.0,
    device="cpu",
    return_matching=True,
    chunk_size=256,
    beta=0.35,
    mode="sparse",
):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)

    for i in range(n1, n):
        Gq.add_node(i)
    for i in range(n2, n):
        Gt.add_node(i)

    F1 = torch.tensor(feature_extraction(Gq), dtype=torch.float64, device=device)
    F2 = torch.tensor(feature_extraction(Gt), dtype=torch.float64, device=device)
    idx, Dk = chunked_topk_support(F1, F2, k=k, chunk_size=chunk_size)

    if mode == "hybrid":
        A_t = torch.tensor(nx.to_numpy_array(Gq), dtype=torch.float64, device=device)
        B_t = torch.tensor(nx.to_numpy_array(Gt), dtype=torch.float64, device=device)
        D_t = eucledian_dist(F1, F2)
        P = FindQuasiPerm_n_k_hybrid(
            A=A_t,
            B=B_t,
            D=D_t,
            mu=mu,
            niter=niter,
            idx=idx,
            reg=reg,
        )
    else:
        A = build_sparse_adjacency(Gq)
        AT = A.transpose().tocsr()
        B_csr = build_sparse_adjacency(Gt)
        B_csc = B_csr.tocsc()
        P = FindQuasiPerm_n_k_sparse(
            A=A,
            AT=AT,
            B_csr=B_csr,
            B_csc=B_csc,
            Dk=Dk,
            idx=idx,
            mu=mu,
            niter=niter,
            reg=reg,
            beta=beta,
        )

    if not return_matching:
        return idx, P

    return sparse_hungarian(idx, P, n1, n2)
