import networkx as nx
import numpy as np
import scipy
import torch

from helpers.pred import feature_extraction


def eucledian_dist(F1, F2):
    """
    Pairwise Euclidean distance on torch tensors.
    """
    if not torch.is_tensor(F1):
        F1 = torch.tensor(F1, dtype=torch.float64)
    if not torch.is_tensor(F2):
        F2 = torch.tensor(F2, dtype=torch.float64)

    return torch.cdist(F1, F2, p=2)


def check_support_cover(idx, n):
    """
    Verify that every real target column appears at least once in the support.
    """
    covered = torch.zeros(n, dtype=torch.bool, device=idx.device)
    covered.scatter_(0, idx.reshape(-1), True)

    missing = (~covered).nonzero(as_tuple=False).reshape(-1)
    if missing.numel() > 0:
        bad = missing.tolist()
        raise ValueError(
            "Support idx does not cover all target columns. "
            f"Missing columns: {bad[:20]}{' ...' if len(bad) > 20 else ''}. "
            "Increase k or change support construction."
        )


def compute_sparse_gradient_n_k(A, B, P, idx, Dk, mu, lam):
    """
    Compute the gradient restricted to the support idx.

    For support entry (i, t) representing real column j = idx[i, t], this
    matches the dense formulation:
        G[i, j] = -(A^T P B)[i, j] - (A P B^T)[i, j] + mu * D[i, j] + lam * (1 - 2P[i, j])
    but is evaluated only on the supported columns.
    """
    n, k = P.shape
    dtype = P.dtype
    device = P.device

    G = torch.empty((n, k), dtype=dtype, device=device)
    unique_js = torch.unique(idx)

    for j_tensor in unique_js:
        j = int(j_tensor.item())

        # x[u] = sum_s P[u, s] * B[idx[u, s], j]
        x = (P * B[idx, j]).sum(dim=1)

        # y[u] = sum_s P[u, s] * B[j, idx[u, s]]
        y = (P * B[j, idx]).sum(dim=1)

        term1 = torch.mv(A.T, x)
        term2 = torch.mv(A, y)

        pos = idx == j
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
    Sinkhorn on the support defined by idx.

    q[i, t] represents the mass on the real entry (i, idx[i, t]).
    """
    n, _ = idx.shape
    dtype = G_sparse.dtype
    device = G_sparse.device
    eps = 1e-12

    if a is None:
        a = torch.ones(n, dtype=dtype, device=device)
    if b is None:
        b = torch.ones(n, dtype=dtype, device=device)

    check_support_cover(idx, n)

    G_shift = G_sparse - G_sparse.min()
    K = torch.exp(-G_shift / reg).clamp_min(eps)

    u = torch.ones(n, dtype=dtype, device=device)
    v = torch.ones(n, dtype=dtype, device=device)

    for _ in range(maxIter):
        u_prev = u
        v_prev = v

        row_sum = (K * v[idx]).sum(dim=1).clamp_min(eps)
        u = a / row_sum

        weighted = u.unsqueeze(1) * K
        col_sum = torch.zeros(n, dtype=dtype, device=device)
        col_sum.scatter_add_(0, idx.reshape(-1), weighted.reshape(-1))
        col_sum = col_sum.clamp_min(eps)
        v = b / col_sum

        du = torch.max(torch.abs(u - u_prev))
        dv = torch.max(torch.abs(v - v_prev))
        if max(float(du), float(dv)) < stopThr:
            break

    return u.unsqueeze(1) * K * v[idx]


def sparse_argmax_matching(idx, P):
    """
    Greedy row-wise decoding on the sparse support.
    """
    best_local = P.argmax(dim=1, keepdim=True)
    return idx.gather(1, best_local).squeeze(1)


def sparse_hungarian(idx, P, n1, n2):
    """
    Support-constrained one-to-one matching without materializing a dense n x n P.
    """
    n = idx.shape[0]
    scores = P.detach().cpu().numpy().reshape(-1)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.detach().cpu().numpy().reshape(-1)

    max_score = float(scores.max()) if scores.size else 0.0
    costs = (max_score - scores) + 1e-12
    sparse_cost = scipy.sparse.coo_matrix((costs, (rows, cols)), shape=(n, n)).tocsr()

    row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(sparse_cost)

    ans = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if i >= n1 or j >= n2:
            continue
        ans.append((i, j))
    return ans


def FindQuasiPerm_n_k(A, B, D, mu, niter, idx, reg=1.0, sinkhorn_iter=500, sinkhorn_thr=1e-5):
    """
    n x k version of FindQuasiPerm that keeps P in sparse-support form throughout.
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device

    Dk = torch.gather(D, 1, idx)
    a = torch.ones(n, dtype=dtype, device=device)
    b = torch.ones(n, dtype=dtype, device=device)

    P = sinkhorn_sparse(
        idx=idx,
        G_sparse=torch.zeros_like(Dk),
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
                lam=outer,
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

    return P


def fly_n_k(Gq, Gt, k, mu=0.5, niter=15, reg=1.0, device="cpu", return_matching=True):
    """
    End-to-end sparse pipeline.

    When return_matching=True, returns the same style output as pred.fly:
        [(query_node, target_node), ...]
    Otherwise returns:
        idx, P
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

    F1 = torch.tensor(feature_extraction(Gq), dtype=torch.float64, device=device)
    F2 = torch.tensor(feature_extraction(Gt), dtype=torch.float64, device=device)

    D = eucledian_dist(F1, F2)
    idx = torch.topk(D, k=k, dim=1, largest=False).indices

    check_support_cover(idx, n)

    P = FindQuasiPerm_n_k(
        A=A,
        B=B,
        D=D,
        mu=mu,
        niter=niter,
        idx=idx,
        reg=reg,
    )

    if not return_matching:
        return idx, P

    return sparse_hungarian(idx, P, n1, n2)
