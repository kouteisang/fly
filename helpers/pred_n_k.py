import torch
import networkx as nx
import numpy as np
from helpers.pred import feature_extraction, eucledian_dist
from sklearn.metrics import pairwise_distances  



def eucledian_dist(F1, F2):
    """
    Pairwise Euclidean distance.
    """
    if not torch.is_tensor(F1):
        F1 = torch.tensor(F1, dtype=torch.float64)
    if not torch.is_tensor(F2):
        F2 = torch.tensor(F2, dtype=torch.float64)

    return torch.cdist(F1, F2, p=2)


def compute_sparse_gradient_n_k(A, B, P, idx, Dk, mu, lam):
    """
    Compute the restricted gradient G on the support defined by idx.

    Inputs
    ------
    A, B : [n, n] dense adjacency matrices
    P    : [n, k]
    idx  : [n, k], idx[i, t] is the real target column for P[i, t]
    Dk   : [n, k], support-restricted distance / cost
    mu   : scalar
    lam  : scalar, corresponds to outer iteration coefficient

    Returns
    -------
    G    : [n, k]
    """
    n, k = P.shape
    dtype = P.dtype
    device = P.device

    G = torch.empty((n, k), dtype=dtype, device=device)

    # Only loop over real target columns that actually appear in idx
    unique_js = torch.unique(idx)

    for j_tensor in unique_js:
        j = int(j_tensor.item())

        # x[u] = sum_s P[u,s] * B[idx[u,s], j]
        Bj_col = B[idx, j]                  # [n, k]
        x = (P * Bj_col).sum(dim=1)         # [n]

        # y[u] = sum_s P[u,s] * B[j, idx[u,s]]
        Bj_row = B[j, idx]                  # [n, k]
        y = (P * Bj_row).sum(dim=1)         # [n]

        # term1[i] = sum_u A[u,i] * x[u] = (A^T x)[i]
        term1 = torch.mv(A.T, x)            # [n]

        # term2[i] = sum_u A[i,u] * y[u] = (A y)[i]
        term2 = torch.mv(A, y)              # [n]

        # Fill only those support positions whose real column is j
        pos = (idx == j)                    # [n, k]
        row_ids = pos.nonzero(as_tuple=True)[0]

        G[pos] = (
            -term1[row_ids]
            -term2[row_ids]
            + mu * Dk[pos]
            + lam * (1.0 - 2.0 * P[pos])
        )

    return G


def sinkhorn_sparse(idx, G_sparse, reg, a=None, b=None, maxIter=500, stopThr=1e-5):
    """
    Sparse masked Sinkhorn on support idx.

    We solve over q in the support defined by idx:
        q[i, t] corresponds to real entry (i, idx[i,t])

    Constraints:
        row sums = a
        real column sums = b

    Inputs
    ------
    idx      : [n, k]
    G_sparse : [n, k]
    reg      : positive scalar
    a        : [n], row marginals, default all ones
    b        : [n], column marginals over real target columns, default all ones

    Returns
    -------
    q        : [n, k]
    """
    n, k = idx.shape
    dtype = G_sparse.dtype
    device = G_sparse.device
    eps = 1e-12

    if a is None:
        a = torch.ones(n, dtype=dtype, device=device)
    if b is None:
        b = torch.ones(n, dtype=dtype, device=device)

    # Feasibility check:
    # if some real column never appears in idx but b[j] > 0, exact masked OT is impossible
    support_count = torch.zeros(n, dtype=dtype, device=device)
    ones_support = torch.ones_like(idx, dtype=dtype, device=device)
    support_count.scatter_add_(0, idx.reshape(-1), ones_support.reshape(-1))

    infeasible_cols = (support_count == 0) & (b > 0)
    if torch.any(infeasible_cols):
        bad = torch.nonzero(infeasible_cols, as_tuple=False).reshape(-1).tolist()
        raise ValueError(
            f"sinkhorn_sparse infeasible: some real target columns are never covered by idx. "
            f"Missing columns: {bad[:20]}{' ...' if len(bad) > 20 else ''}. "
            f"Increase k or change support construction."
        )

    # Kernel
    # Global shift improves numerical stability and does not change the Sinkhorn solution
    G_shift = G_sparse - G_sparse.min()
    K = torch.exp(-G_shift / reg).clamp_min(eps)      # [n, k]

    u = torch.ones(n, dtype=dtype, device=device)
    v = torch.ones(n, dtype=dtype, device=device)

    for _ in range(maxIter):
        u_prev = u
        v_prev = v

        # Row update:
        # u[i] = a[i] / sum_t K[i,t] * v[idx[i,t]]
        Kv = K * v[idx]
        row_sum = Kv.sum(dim=1).clamp_min(eps)
        u = a / row_sum

        # Column update:
        # col_sum[j] = sum_{i,t: idx[i,t]=j} u[i] * K[i,t]
        weighted = (u.unsqueeze(1) * K)               # [n, k]
        col_sum = torch.zeros(n, dtype=dtype, device=device)
        col_sum.scatter_add_(0, idx.reshape(-1), weighted.reshape(-1))
        col_sum = col_sum.clamp_min(eps)

        v = b / col_sum

        du = torch.max(torch.abs(u - u_prev))
        dv = torch.max(torch.abs(v - v_prev))
        if max(float(du), float(dv)) < stopThr:
            break

    q = u.unsqueeze(1) * K * v[idx]                  # [n, k]
    return q


def FindQuasiPerm_n_k(A, B, D, mu, niter, idx, k, reg=1.0, sinkhorn_iter=500, sinkhorn_thr=1e-5):
    """
    n x k version of FindQuasiPerm.

    This preserves the original outer structure:
        G <- gradient-like matrix on support
        q <- Sinkhorn(G)
        P <- P + alpha * (q - P)

    Inputs
    ------
    A, B : [n, n]
    D    : [n, n]
    idx  : [n, k]
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device
    eps = 1e-12

    Dk = torch.gather(D, 1, idx)                     # [n, k]

    a = torch.ones(n, dtype=dtype, device=device)
    b = torch.ones(n, dtype=dtype, device=device)

    # Masked analogue of the original doubly-stochastic uniform initialization
    G0 = torch.zeros((n, k), dtype=dtype, device=device)
    P = sinkhorn_sparse(
        idx=idx,
        G_sparse=G0,
        reg=reg,
        a=a,
        b=b,
        maxIter=sinkhorn_iter,
        stopThr=sinkhorn_thr,
    )

    for outer in range(niter):
        for it in range(1, 11):
            G = compute_sparse_gradient_n_k(
                A=A,
                B=B,
                P=P,
                idx=idx,
                Dk=Dk,
                mu=mu,
                lam=outer
            )

            q = sinkhorn_sparse(
                idx=idx,
                G_sparse=G,
                reg=reg,
                a=a,
                b=b,
                maxIter=sinkhorn_iter,
                stopThr=sinkhorn_thr,
            )

            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)

            # Small cleanup for numerical stability
            P = P.clamp_min(eps)

            # Row normalize
            P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)

            # Sparse real-column rebalance
            col_sum = torch.zeros(n, dtype=dtype, device=device)
            col_sum.scatter_add_(0, idx.reshape(-1), P.reshape(-1))
            scale = 1.0 / col_sum[idx].clamp_min(eps)
            P = P * scale

            # Row normalize again
            P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)

    return P


def sparse_matching_from_P(idx, P):
    """
    Hard matching:
        match[i] = real target column chosen for row i
    """
    best_local = P.argmax(dim=1, keepdim=True)       # [n, 1]
    match = idx.gather(1, best_local).squeeze(1)     # [n]
    return match


def fly_n_k(Gq, Gt, k=k, mu=0.5, niter=5, reg=1.0, device="cpu"):

    """
    End-to-end pipeline.

    Returns
    -------
    idx : [n, k]
    P   : [n, k]
    """
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)

    for i in range(n1, n):
        Gq.add_node(i)
    for i in range(n2, n):
        Gt.add_node(i)

    A = torch.tensor(nx.to_numpy_array(Gq), dtype=torch.float64, device=device)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype=torch.float64, device=device)

    F1 = feature_extraction(Gq).to(device)
    F2 = feature_extraction(Gt).to(device)

    D = eucledian_dist(F1, F2).to(device)

    idx = torch.topk(D, k=k, dim=1, largest=False).indices

    P = FindQuasiPerm_n_k(
        A=A,
        B=B,
        D=D,
        mu=mu,
        niter=niter,
        idx=idx,
        k=k,
        reg=reg
    )

    print(P.shape)
    print(P.sum(dim=1)[:10])
    print(sparse_matching_from_P(idx, P)[:10])

    return idx, P
