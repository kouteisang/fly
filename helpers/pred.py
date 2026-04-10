import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import torch
from helpers import sinkhorn
import scipy

def convertToPermHungarian(M, n1, n2):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    n = len(M)

    P = np.zeros((n, n))
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans


def feature_extraction(G):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts

    node_features = np.nan_to_num(node_features)
    return np.nan_to_num(node_features)

def eucledian_dist(F1, F2, n):
    D = euclidean_distances(F1, F2)
    return D


def FindQuasiPerm(A, B, D, mu, niter, idx, k):
    n = A.shape[0]
    dtype = torch.float64
    device = A.device
    eps = 1e-12
    
    P = torch.zeros((n, n), dtype=torch.float64)
    P.scatter_(1, idx, 1.0 / k)
    
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K = mu * D

    mask = torch.zeros((n, n), dtype=dtype, device=device)
    mask.scatter_(1, idx, 1.0)

    for i in range(niter):
        for it in range(1, 11):
            G = -torch.mm(torch.mm(A.T, P), B) - torch.mm(torch.mm(A, P), B.T) + K + i*(mat_ones - 2*P)
            q = sinkhorn.sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-5)
            # alpha = 2.0 / float(2.0 + it)
            # P = P + alpha * (q - P)
            # 5. 更新后再次投影回 mask
            q = q * mask
            q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
            
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)

            P = P * mask
            P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)
    return P


def fly(Gq, Gt, n, k, mu=0.5, niter=15):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)

    for i in range(n1, n):
        Gq.add_node(i)
    for i in range(n2, n):
        Gt.add_node(i)

    A = torch.tensor(nx.to_numpy_array(Gq), dtype=torch.float64)
    B = torch.tensor(nx.to_numpy_array(Gt), dtype=torch.float64)

    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)

    if not torch.is_tensor(F1):
        F1 = torch.tensor(F1, dtype=torch.float64)
    if not torch.is_tensor(F2):
        F2 = torch.tensor(F2, dtype=torch.float64)

    D = torch.cdist(F1, F2, p=2)

    idx = torch.topk(D, k=k, dim=1, largest=False).indices

    P = FindQuasiPerm(A, B, D, mu, niter, idx, k)
    
    P_perm, ans = convertToPermHungarian(P, n, n)
    return ans
