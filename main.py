import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np

from helpers.metrics import eval_align
from helpers.pred import fly
from helpers.pred_lowrank import fly_lowrank
from helpers.pred_n_k import fly_n_k


ROOT = Path(__file__).resolve().parent
DEFAULT_QUERY = "data/real_noise/contacts-prox-high-school-2013/contacts-prox-high-school-2013_100.txt"
DEFAULT_TARGET = "data/real_noise/contacts-prox-high-school-2013/contacts-prox-high-school-2013_80.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Run graph alignment experiments.")
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Path to the query graph edge list, relative to repo root or absolute.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Path to the target graph edge list, relative to repo root or absolute.",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid", "lowrank"],
        default="dense",
        help="Choose dense, true sparse, hybrid, or lowrank alignment.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional total node count. If omitted, infer from the input files.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Top-k candidate count per query node. Default is n//2.",
    )
    parser.add_argument("--mu", type=float, default=0.5, help="Feature cost weight.")
    parser.add_argument("--niter", type=int, default=15, help="Outer iteration count.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the sparse implementation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size used when building sparse top-k supports.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.35,
        help="Soft column rebalance strength for the true sparse transport update.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="Low-rank embedding dimension for lowrank mode.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Distance temperature used to convert lowrank distances into scores.",
    )
    parser.add_argument(
        "--feature-weight",
        type=float,
        default=1.0,
        help="Weight of handcrafted features in lowrank mode.",
    )
    parser.add_argument(
        "--spectral-weight",
        type=float,
        default=1.0,
        help="Weight of spectral embeddings in lowrank mode.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def infer_node_count_from_edges(path):
    max_node = -1
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            u_str, v_str = line.split()
            max_node = max(max_node, int(u_str), int(v_str))
    return max_node + 1


def load_graph(path, n=None):
    graph = nx.Graph()

    if n is None:
        n = infer_node_count_from_edges(path)

    graph.add_nodes_from(range(n))
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            u_str, v_str = line.split()
            graph.add_edge(int(u_str), int(v_str))
    return graph


def run_alignment(args, Gquery, Gtarget, n, k):
    if args.mode == "dense":
        return fly(Gquery, Gtarget, n, k, mu=args.mu, niter=args.niter)

    if args.mode == "lowrank":
        return fly_lowrank(
            Gquery,
            Gtarget,
            k=k,
            rank=args.rank,
            chunk_size=args.chunk_size,
            tau=args.tau,
            feature_weight=args.feature_weight,
            spectral_weight=args.spectral_weight,
        )

    return fly_n_k(
        Gquery,
        Gtarget,
        k=k,
        mu=args.mu,
        niter=args.niter,
        device=args.device,
        return_matching=True,
        chunk_size=args.chunk_size,
        beta=args.beta,
        mode=args.mode,
    )


def main():
    args = parse_args()

    query_path = resolve_path(args.query)
    target_path = resolve_path(args.target)

    inferred_n = max(
        infer_node_count_from_edges(query_path),
        infer_node_count_from_edges(target_path),
    )
    n = args.n if args.n is not None else inferred_n
    k = args.k if args.k is not None else n // 2

    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}.")

    Gquery = load_graph(query_path, n=n)
    Gtarget = load_graph(target_path, n=n)

    gmb = np.arange(Gtarget.number_of_nodes())

    time_start = time.time()
    ans = run_alignment(args, Gquery, Gtarget, n, k)
    time_end = time.time()

    ma = np.array([pair[0] for pair in ans], dtype=int)
    mb = np.array([pair[1] for pair in ans], dtype=int)

    gacc, acc, _ = eval_align(ma, mb, gmb)

    print(f"mode: {args.mode}")
    print(f"query: {query_path}")
    print(f"target: {target_path}")
    print(f"n: {n}")
    print(f"k: {k}")
    print(f"mu: {args.mu}")
    print(f"niter: {args.niter}")
    print(f"beta: {args.beta}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"rank: {args.rank}")
    print(f"tau: {args.tau}")
    print(f"feature_weight: {args.feature_weight}")
    print(f"spectral_weight: {args.spectral_weight}")
    print(f"time: {time_end - time_start}")
    print(f"matched_pairs: {len(ans)}")
    print(f"gacc: {gacc}")
    print(f"acc: {acc}")


if __name__ == "__main__":
    main()
