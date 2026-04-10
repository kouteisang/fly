import networkx as nx
import numpy as np
import scipy
import torch

from helpers.pred import feature_extraction
from helpers import sinkhorn


def eucledian_dist(F1, F2):
    """
    Pairwise Euclidean distance on torch tensors.
    """
    if not torch.is_tensor(F1):
        F1 = torch.tensor(F1, dtype=torch.float64)
    if not torch.is_tensor(F2):
        F2 = torch.tensor(F2, dtype=torch.float64)

    return torch.cdist(F1, F2, p=2)


def sparse_to_dense(idx, P_sparse, n):
    """
    Materialize a dense n x n matrix from the sparse support representation.
    """
    P_dense = torch.zeros((n, n), dtype=P_sparse.dtype, device=P_sparse.device)
    P_dense.scatter_(1, idx, P_sparse)
    return P_dense


def dense_to_sparse(idx, M_dense):
    """
    Gather supported entries from a dense matrix into n x k form.
    """
    return torch.gather(M_dense, 1, idx)


def sparse_argmax_matching(idx, P):
    """
    Greedy row-wise decoding on the sparse support.
    """
    best_local = P.argmax(dim=1, keepdim=True)
    return idx.gather(1, best_local).squeeze(1)


def sparse_hungarian(idx, P, n1, n2):
    """
    Dense Hungarian with implicit zero completion outside the sparse support.
    """
    n = idx.shape[0]
    score_matrix = np.zeros((n, n), dtype=np.float64)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.detach().cpu().numpy().reshape(-1)
    score_matrix[rows, cols] = P.detach().cpu().numpy().reshape(-1)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(score_matrix, maximize=True)

    ans = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if i >= n1 or j >= n2:
            continue
        ans.append((i, j))
    return ans


def FindQuasiPerm_n_k(A, B, D, mu, niter, idx, reg=1.0, sinkhorn_iter=500, sinkhorn_thr=1e-5):
    """
    Hybrid version:
    P is stored as n x k, but each q update follows the dense pred.py semantics
    exactly by temporarily materializing the masked dense matrix.
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device
    eps = 1e-12
    k = idx.shape[1]

    ones = torch.ones(n, dtype=dtype, device=device)
    mat_ones = torch.ones((n, n), dtype=dtype, device=device)
    P = torch.full((n, k), 1.0 / k, dtype=dtype, device=device)

    for outer in range(niter):
        for it in range(1, 11):
            P_dense = sparse_to_dense(idx, P, n)
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
            q = dense_to_sparse(idx, q_dense)
            q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)

            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
            P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)

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
