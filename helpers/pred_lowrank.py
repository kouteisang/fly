import networkx as nx
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh

from helpers.pred import feature_extraction


def build_sparse_adjacency(G):
    adj = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
    return scipy.sparse.csr_matrix(adj)


def standardize(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (X - mean) / std


def truncated_spectral_embedding(G, rank):
    A = build_sparse_adjacency(G)
    n = A.shape[0]
    rank = max(1, min(rank, n - 2))

    vals, vecs = eigsh(A, k=rank, which="LM")
    order = np.argsort(-np.abs(vals))
    vals = vals[order]
    vecs = vecs[:, order]

    emb = vecs * np.sqrt(np.abs(vals) + 1e-12)
    deg = np.asarray(A.sum(axis=1)).ravel()

    for col in range(emb.shape[1]):
        if np.dot(emb[:, col], deg) < 0:
            emb[:, col] *= -1.0

    return emb


def chunked_top1(source, target, chunk_size=256):
    inds = []
    dists = []
    for start in range(0, source.shape[0], chunk_size):
        end = min(start + chunk_size, source.shape[0])
        D = scipy.spatial.distance.cdist(source[start:end], target, metric="euclidean")
        best = D.argmin(axis=1)
        inds.append(best)
        dists.append(D[np.arange(end - start), best])
    return np.concatenate(inds), np.concatenate(dists)


def mutual_feature_pairs(F1, F2, chunk_size=256, max_pairs=256):
    nn12, d12 = chunked_top1(F1, F2, chunk_size=chunk_size)
    nn21, _ = chunked_top1(F2, F1, chunk_size=chunk_size)

    pairs = []
    for i, j in enumerate(nn12):
        if nn21[j] == i:
            pairs.append((i, int(j), float(d12[i])))

    if not pairs:
        order = np.argsort(d12)
        keep = order[: min(max_pairs, F1.shape[0])]
        return [(int(i), int(nn12[i])) for i in keep]

    pairs.sort(key=lambda x: x[2])
    pairs = pairs[: min(max_pairs, len(pairs))]
    return [(i, j) for i, j, _ in pairs]


def orthogonal_align(X, Y, pairs):
    if not pairs:
        return np.eye(X.shape[1], dtype=np.float64)

    src = np.stack([X[i] for i, _ in pairs], axis=0)
    tgt = np.stack([Y[j] for _, j in pairs], axis=0)
    M = src.T @ tgt
    U, _, VT = np.linalg.svd(M, full_matrices=False)
    return U @ VT


def chunked_topk_scores(Z1, Z2, k, chunk_size=256, tau=1.0):
    idx_chunks = []
    score_chunks = []
    for start in range(0, Z1.shape[0], chunk_size):
        end = min(start + chunk_size, Z1.shape[0])
        D = scipy.spatial.distance.cdist(Z1[start:end], Z2, metric="euclidean")
        vals, inds = topk_smallest(D, k)
        scores = np.exp(-vals / max(tau, 1e-12))
        idx_chunks.append(inds)
        score_chunks.append(scores)
    return np.vstack(idx_chunks), np.vstack(score_chunks)


def topk_smallest(D, k):
    part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(D, part, axis=1)
    order = np.argsort(vals, axis=1)
    inds = np.take_along_axis(part, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return vals, inds


def sparse_hungarian(idx, scores, n1, n2):
    n, k = idx.shape
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    weights = scores.reshape(-1)

    max_score = float(weights.max()) if weights.size else 0.0
    costs = (max_score - weights) + 1e-12
    graph = scipy.sparse.coo_matrix((costs, (rows, cols)), shape=(n, n)).tocsr()

    try:
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(graph)
        ans = []
        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            if i < n1 and j < n2:
                ans.append((i, j))
        return ans
    except Exception:
        order = np.argsort(-weights, kind="stable")
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


def fly_lowrank(
    Gq,
    Gt,
    k,
    rank=32,
    chunk_size=256,
    tau=1.0,
    feature_weight=1.0,
    spectral_weight=1.0,
):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)

    for i in range(n1, n):
        Gq.add_node(i)
    for i in range(n2, n):
        Gt.add_node(i)

    F1 = standardize(feature_extraction(Gq))
    F2 = standardize(feature_extraction(Gt))

    E1 = standardize(truncated_spectral_embedding(Gq, rank))
    E2 = standardize(truncated_spectral_embedding(Gt, rank))

    pairs = mutual_feature_pairs(F1, F2, chunk_size=chunk_size, max_pairs=max(64, 4 * rank))
    R = orthogonal_align(E1, E2, pairs)
    E1_aligned = E1 @ R

    Z1 = np.concatenate([feature_weight * F1, spectral_weight * E1_aligned], axis=1)
    Z2 = np.concatenate([feature_weight * F2, spectral_weight * E2], axis=1)
    Z1 = standardize(Z1)
    Z2 = standardize(Z2)

    idx, scores = chunked_topk_scores(Z1, Z2, k=k, chunk_size=chunk_size, tau=tau)
    return sparse_hungarian(idx, scores, n1, n2)
