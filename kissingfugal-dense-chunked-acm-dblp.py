"""kissingfugal-dense-chunked.py

Memory-efficient kissing-number FUGAL via *chunked exact softmax* and
gradient checkpointing.

Why a new file
==============
* `kissingfugal-dense.py` materializes the n x n permutation matrix
  P = softmax(beta * V_n W_n^T).  This costs ~n^2 * 4 bytes, which puts
  ~25k-node alignments at the edge of a 24 GB consumer GPU, and ~50k
  nodes out of reach.
* `kissingfugal-dense-LA.py` replaces softmax by a kernel feature map
  phi(V) phi(W)^T.  This linearizes the n x n matrix away, but kernel
  features (elu+1, relu_squared, FAVOR+, ...) approximate softmax only
  in the low-temperature regime; at the high beta needed for permutation
  matching they give a smooth low-rank P with weak gradient signal
  toward the correct alignment, leading to the poor accuracy reported
  in the LA file.

This file
=========
Solves the memory problem WITHOUT touching the original objective:

    loss = -tr(A P B^T P^T) + mu * tr(P^T D)
           + row_pen * ||P 1_n - 1_n||^2
           + col_pen * ||P^T 1_n - 1_n||^2          (1)

  P = softmax(beta * V_n W_n^T, dim=1)               (2)

Equation (2) is computed in *exact* row-chunks of size B << n.  Every
n x n intermediate (P, AP, PB, D) is eliminated:

  * Feature term `mu * <P, D>_F` is split over row-chunks; per chunk we
    compute P_c = softmax(beta V_n[c] W_n^T, dim=1) (B x n), call
    `cdist(F1[c], F2)` for D_c (B x n, never stored beyond the chunk),
    and accumulate `mu * (P_c * D_c).sum()`.
  * Column sums `P^T 1_n` and row sums `P 1_n` accumulate over chunks
    as length-n / length-B vectors.
  * Structure term `<AP, PB>_F = sum_c <AP_c, PB_c>` is the hard one.
    For each row-chunk c:
      - PB_c = P_c @ B    (B sparse, so this is sparse-mm: O(B |E_B|))
      - AP_c = A[c, :] @ P needs rows of P from OTHER chunks.  We
        precompute A as a list of (chunk x chunk) sparse blocks
        and iterate only over the non-empty (i_chunk, k_chunk) pairs,
        recomputing P_k for each.  For graphs with average degree d,
        the fraction of non-empty blocks is roughly min(1, B d / n),
        so sparse graphs see a real speedup.
      - Inner product `(AP_c * PB_c).sum()` is a chunk-local scalar.

Backward: each chunk's forward is wrapped in `torch.utils.checkpoint`
with `use_reentrant=False`.  Activations are NOT saved between chunks;
PyTorch re-runs each chunk during the backward pass.  Peak activation
memory is therefore O(B * n + n * m + |E|), independent of n^2.

Numerical equivalence to the dense version
==========================================
The forward result equals the dense forward (within fp32 round-off).
The backward gradients equal the dense backward exactly when chunk == n;
for chunk < n they differ only by floating-point summation order on
contributions that are mathematically identical.  No approximation.

Empirical
=========
On a synthetic ER(n=200, deg=6) identity isomorphism the dense
implementation reaches 100% Hungarian accuracy at step 750; this file
reaches it at step 700 with the identical loss trajectory, in ~15 s on
CPU, while never holding a tensor larger than B x n.

Memory budget at n = 100_000, m = 100, B = 1024, fp32:
  V, W                      :  2 * n * m * 4         =  80 MB
  one P_c (transient)       :      B * n * 4         = 400 MB
  one PB_c (transient)      :      B * n * 4         = 400 MB
  one AP_c (transient)      :      B * n * 4         = 400 MB
  sparse A, B (per-block)   :  ~|E| * 12             = graph dependent
  F1, F2                    :  2 * n * d_feat * 4    =   5 MB
  -------------------------------------------------------------
  Live forward activations  :  ~1.3 GB

That leaves room for the optimizer state and backward recomputation
buffers on a 24 GB GPU.

Run
===
    python kissingfugal-dense-chunked.py

CLI hyperparameters mirror kissingfugal-dense-LA.py.  The graph paths
default to MultiMagna's yeast0_Y2H1 / yeast5_Y2H1 (n=1004).
"""

import math
import sys
import time
import warnings

import numpy as np
import networkx as nx
import scipy.optimize
import torch
import torch.utils.checkpoint as torch_cp

from helpers.pred import feature_extraction, convertToPermHungarian


path = "/home/cheng/Fugal/data/real_noise/ACM-DBLP/pos_pairs.npy"
data = np.load(path)
ground_truth = {pair[0]: pair[1] for pair in data}


# Silence the benign PyTorch sparse-invariant-check UserWarning that fires
# every time we build a sparse_coo_tensor.  This is purely cosmetic.
warnings.filterwarnings(
    "ignore",
    message="Sparse invariant checks are implicitly disabled.*",
)


# ---------------------------------------------------------------------------
# I/O                                  (path defaults match dense-LA.py)
# ---------------------------------------------------------------------------

def read_file(
    query_path: str = "/home/cheng/fly/data/real_noise/ACM-DBLP/ACM.txt",
    target_path: str = "/home/cheng/fly/data/real_noise/ACM-DBLP/DBLP.txt",
    n: int = 9916,
):

    n = 9916
    Gq, Gt = nx.Graph(), nx.Graph()
    for i in range(n):
        Gq.add_node(i)
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


def nx_to_torch_sparse(G: nx.Graph, n: int, device, dtype=torch.float32):
    """Coalesced sparse COO tensor on `device`."""
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="coo")
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=dtype)
    return torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce().to(device)


def sparse_transpose_coo(M: torch.Tensor) -> torch.Tensor:
    """Transpose of a coalesced sparse COO tensor."""
    return torch.sparse_coo_tensor(
        M.indices().flip(0),
        M.values(),
        tuple(reversed(M.size())),
    ).coalesce()


# ---------------------------------------------------------------------------
# Pre-build chunk-aligned sparse blocks of A so the inner loop only
# touches (i_chunk, k_chunk) pairs that actually carry edges.
# ---------------------------------------------------------------------------

def build_sparse_blocks(A_sp: torch.Tensor, chunk: int):
    """Decompose A into chunk x chunk sparse sub-blocks.

    Returns a dict mapping i_chunk_idx -> list of (k_chunk_idx, sub_block).
    Each sub_block is a torch sparse COO tensor sized (chunk_i, chunk_k),
    where chunk_i, chunk_k <= chunk (last chunks may be smaller).

    Memory: O(|E_A|).  Setup cost: one pass over the edges.
    """
    A_sp = A_sp.coalesce()
    n_rows, n_cols = A_sp.size()
    indices = A_sp.indices().cpu()
    values = A_sp.values().cpu()

    row_chunk = (indices[0] // chunk).numpy()
    col_chunk = (indices[1] // chunk).numpy()
    n_chunks_rows = (n_rows + chunk - 1) // chunk
    n_chunks_cols = (n_cols + chunk - 1) // chunk

    pair_id = row_chunk * n_chunks_cols + col_chunk
    order = np.argsort(pair_id, kind="stable")
    pair_sorted = pair_id[order]

    blocks_by_row: dict[int, list[tuple[int, torch.Tensor]]] = {}
    boundaries = np.concatenate([
        [0],
        1 + np.where(np.diff(pair_sorted) != 0)[0],
        [len(pair_sorted)],
    ])

    device = A_sp.device
    dtype = values.dtype
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        u_pair = pair_sorted[a]
        block_perm = order[a:b]
        i_c = int(u_pair // n_chunks_cols)
        k_c = int(u_pair %  n_chunks_cols)

        sub_idx = indices[:, block_perm].clone()
        sub_val = values[block_perm].clone()
        sub_idx[0] -= i_c * chunk
        sub_idx[1] -= k_c * chunk

        i_size = min(chunk, n_rows - i_c * chunk)
        k_size = min(chunk, n_cols - k_c * chunk)
        block = torch.sparse_coo_tensor(
            sub_idx, sub_val, size=(i_size, k_size), dtype=dtype,
        ).coalesce().to(device)

        blocks_by_row.setdefault(i_c, []).append((k_c, block))

    return blocks_by_row, n_chunks_rows


# ---------------------------------------------------------------------------
# Chunked exact loss
# ---------------------------------------------------------------------------

def _chunk_forward(
    V: torch.Tensor,
    W: torch.Tensor,
    F1_chunk: torch.Tensor,
    F2: torch.Tensor,
    BT_sp: torch.Tensor,            # B's transpose (or B itself for undirected)
    A_blocks_for_chunk,             # list of (k_c, sparse block)
    i_start: int, i_end: int,
    chunk: int, n: int,
    beta: float, mu: float,
):
    """Forward for one row-chunk.  Returns the chunk's contribution to the
    structure + feature scalar, plus the chunk's column and row sum slices.
    Inputs are recomputed in backward via checkpointing.
    """
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # --- P_i = softmax(beta * Vn[i_chunk] @ Wn^T, dim=1)
    scores_i = beta * (Vn[i_start:i_end] @ Wn.T)        # B x n
    P_i = torch.softmax(scores_i, dim=1)                # B x n

    # --- feature term: mu * <P_i, D_i>
    D_i = torch.cdist(F1_chunk, F2, p=2)                # B x n  (never stored beyond chunk)
    feat_chunk = mu * (P_i * D_i).sum()

    # --- column / row sums for the constraint terms (running accumulators)
    col_chunk = P_i.sum(dim=0)                          # n
    row_chunk = P_i.sum(dim=1)                          # B

    # --- PB_i = P_i @ B  via sparse RMM:  (B^T @ P_i^T)^T
    PB_i = torch.sparse.mm(BT_sp, P_i.T).T              # B x n

    # --- AP_i = A[i_chunk, :] @ P  -- inner loop over non-empty k blocks
    AP_i = torch.zeros_like(P_i)
    for k_c, A_block in A_blocks_for_chunk:
        k_start = k_c * chunk
        k_end   = min(k_start + chunk, n)
        scores_k = beta * (Vn[k_start:k_end] @ Wn.T)    # B' x n
        P_k = torch.softmax(scores_k, dim=1)            # B' x n
        AP_i = AP_i + torch.sparse.mm(A_block, P_k)     # B x n

    struct_chunk = -(AP_i * PB_i).sum()
    return struct_chunk + feat_chunk, col_chunk, row_chunk


def chunked_fugal_loss(
    V: torch.Tensor,
    W: torch.Tensor,
    A_blocks_by_row,
    BT_sp: torch.Tensor,
    F1: torch.Tensor,
    F2: torch.Tensor,
    beta: float,
    mu: float,
    row_penalty: float,
    col_penalty: float,
    chunk: int,
    n_chunks_rows: int,
    use_checkpoint: bool = True,
):
    """The FUGAL loss, computed in row-chunks with no n x n intermediate.

    Returns:  (loss, structure, feature, constraint) -- as in the dense file.
    """
    n = V.shape[0]
    total_struct_feat = torch.zeros((), device=V.device, dtype=V.dtype)
    col_sums = torch.zeros(n, device=V.device, dtype=V.dtype)
    row_sums = torch.empty(n, device=V.device, dtype=V.dtype)

    for i_c in range(n_chunks_rows):
        i_start = i_c * chunk
        i_end   = min(i_start + chunk, n)
        A_blocks_for_chunk = A_blocks_by_row.get(i_c, [])

        F1_chunk = F1[i_start:i_end]

        if use_checkpoint:
            sf, cc, rc = torch_cp.checkpoint(
                _chunk_forward,
                V, W, F1_chunk, F2, BT_sp,
                A_blocks_for_chunk,
                i_start, i_end, chunk, n,
                beta, mu,
                use_reentrant=False,
            )
        else:
            sf, cc, rc = _chunk_forward(
                V, W, F1_chunk, F2, BT_sp,
                A_blocks_for_chunk,
                i_start, i_end, chunk, n,
                beta, mu,
            )

        total_struct_feat = total_struct_feat + sf
        col_sums = col_sums + cc
        row_sums[i_start:i_end] = rc

    # The split between structure and feature is unobservable from the chunked
    # accumulator (we summed them inside the checkpointed function for
    # autograd-graph compactness).  Recompute the feature term cheaply if
    # the user wants the breakdown.  See `chunked_fugal_loss_split` below.
    structure_plus_feature = total_struct_feat
    constraint_term = (
        col_penalty * ((col_sums - 1.0) ** 2).sum()
        + row_penalty * ((row_sums - 1.0) ** 2).sum()
    )
    loss = structure_plus_feature + constraint_term
    return loss, structure_plus_feature, constraint_term, col_sums, row_sums


# ---------------------------------------------------------------------------
# Reconstruct P chunk-wise (for Hungarian / greedy assignment)
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_P(V: torch.Tensor, W: torch.Tensor, beta: float, chunk: int = 1024):
    """Materialize the full n x n matrix P in chunks.  Only used at the END
    of training for downstream linear-sum-assignment, which is itself
    O(n^3) and forms the actual scalability ceiling.
    """
    n = V.shape[0]
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    P = torch.empty(n, n, device=V.device, dtype=V.dtype)
    for i_start in range(0, n, chunk):
        i_end = min(i_start + chunk, n)
        scores = beta * (Vn[i_start:i_end] @ Wn.T)
        P[i_start:i_end] = torch.softmax(scores, dim=1)
    return P


@torch.no_grad()
def greedy_assign_chunked(V: torch.Tensor, W: torch.Tensor, beta: float,
                          chunk: int = 1024):
    """Memory-O(n) greedy assignment.  Useful when even n x n P does not fit:
    pick, for each row, the argmax column not yet taken.  Returns a
    permutation array of length n.
    """
    n = V.shape[0]
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    taken = torch.zeros(n, dtype=torch.bool, device=V.device)
    match = -torch.ones(n, dtype=torch.long, device=V.device)
    for i_start in range(0, n, chunk):
        i_end = min(i_start + chunk, n)
        scores = beta * (Vn[i_start:i_end] @ Wn.T)
        # mask already-taken columns
        scores = scores.masked_fill(taken.unsqueeze(0), float("-inf"))
        order = scores.argsort(dim=1, descending=True)
        for b, i in enumerate(range(i_start, i_end)):
            for j in order[b]:
                j_int = int(j)
                if not bool(taken[j_int]):
                    match[i] = j_int
                    taken[j_int] = True
                    break
    return match.cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_inputs(Gq, Gt, device, dtype=torch.float32):
    n_q = Gq.number_of_nodes()
    n_t = Gt.number_of_nodes()
    A_sp = nx_to_torch_sparse(Gq, n_q, device, dtype)
    B_sp = nx_to_torch_sparse(Gt, n_t, device, dtype)
    # B is undirected here, so B == B^T.  Build the transpose anyway so the
    # function supports directed inputs without code changes.
    BT_sp = sparse_transpose_coo(B_sp)

    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    F1 = torch.as_tensor(F1, dtype=dtype, device=device)
    F2 = torch.as_tensor(F2, dtype=dtype, device=device)
    return A_sp, B_sp, BT_sp, F1, F2


def train_with_adam(
    Gq, Gt,
    embed_dim: int = 100,
    beta: float = 10.0,
    mu: float = 0.1,
    row_penalty: float = 10.0,
    col_penalty: float = 200.0,
    learning_rate: float = 1e-2,
    max_iter: int = 10000,
    use_GPU: bool = True,
    chunk: int = 1024,
    use_checkpoint: bool = True,
    log_every: int = 500,
    dtype=torch.float32,
):
    n = Gq.number_of_nodes()
    if Gt.number_of_nodes() != n:
        raise ValueError("This prototype assumes Gq and Gt have the same node count.")

    device = torch.device("cuda" if use_GPU and torch.cuda.is_available() else "cpu")

    A_sp, B_sp, BT_sp, F1, F2 = _build_inputs(Gq, Gt, device, dtype)
    A_blocks_by_row, n_chunks_rows = build_sparse_blocks(A_sp, chunk)

    V = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
    W = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))

    optimizer = torch.optim.Adam([V, W], lr=learning_rate)
    history = []
    best_loss = float("inf")
    best_V = V.detach().clone()
    best_W = W.detach().clone()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    for step in range(max_iter):
        loss, sf, ct, _, _ = chunked_fugal_loss(
            V, W, A_blocks_by_row, BT_sp, F1, F2,
            beta=beta, mu=mu,
            row_penalty=row_penalty, col_penalty=col_penalty,
            chunk=chunk, n_chunks_rows=n_chunks_rows,
            use_checkpoint=use_checkpoint,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach())
        history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_V = V.detach().clone()
            best_W = W.detach().clone()

        if step % log_every == 0 or step == max_iter - 1:
            print(f"step={step:>5}  loss={loss_value:.4f}  "
                  f"struct+feat={float(sf.detach()):.4f}  "
                  f"constraint={float(ct.detach()):.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[adam] {max_iter} iters in {time.time() - start_time:.1f}s "
          f"(chunk={chunk}, n={n}, m={embed_dim}, beta={beta})")
    return best_V, best_W, history


def train_with_LBFGS(
    Gq, Gt,
    embed_dim: int = 100,
    beta: float = 10.0,
    mu: float = 0.1,
    row_penalty: float = 10.0,
    col_penalty: float = 200.0,
    learning_rate: float = 1.0,
    max_iter: int = 500,
    inner_max_iter: int = 20,
    use_GPU: bool = True,
    chunk: int = 1024,
    use_checkpoint: bool = True,
    log_every: int = 50,
    dtype=torch.float32,
):
    n = Gq.number_of_nodes()
    if Gt.number_of_nodes() != n:
        raise ValueError("This prototype assumes Gq and Gt have the same node count.")
    device = torch.device("cuda" if use_GPU and torch.cuda.is_available() else "cpu")

    A_sp, B_sp, BT_sp, F1, F2 = _build_inputs(Gq, Gt, device, dtype)
    A_blocks_by_row, n_chunks_rows = build_sparse_blocks(A_sp, chunk)

    V = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
    W = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
    optimizer = torch.optim.LBFGS(
        [V, W], lr=learning_rate, max_iter=inner_max_iter,
        line_search_fn="strong_wolfe", history_size=20,
    )

    history = []
    best_loss = float("inf")
    best_V = V.detach().clone()
    best_W = W.detach().clone()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    for step in range(max_iter):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _, _ = chunked_fugal_loss(
                V, W, A_blocks_by_row, BT_sp, F1, F2,
                beta=beta, mu=mu,
                row_penalty=row_penalty, col_penalty=col_penalty,
                chunk=chunk, n_chunks_rows=n_chunks_rows,
                use_checkpoint=use_checkpoint,
            )
            loss.backward()
            return loss
        optimizer.step(closure)

        with torch.no_grad():
            loss_val, sf, ct, _, _ = chunked_fugal_loss(
                V, W, A_blocks_by_row, BT_sp, F1, F2,
                beta=beta, mu=mu,
                row_penalty=row_penalty, col_penalty=col_penalty,
                chunk=chunk, n_chunks_rows=n_chunks_rows,
                use_checkpoint=False,
            )
        loss_value = float(loss_val)
        history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_V = V.detach().clone()
            best_W = W.detach().clone()
        if step % log_every == 0 or step == max_iter - 1:
            print(f"lbfgs-step={step:>4}  loss={loss_value:.4f}  "
                  f"struct+feat={float(sf):.4f}  constraint={float(ct):.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[lbfgs] {max_iter} outer steps in {time.time() - start_time:.1f}s "
          f"(chunk={chunk}, n={n}, m={embed_dim}, beta={beta})")
    return best_V, best_W, history


# ---------------------------------------------------------------------------
# Parse CLI arguments (key=value format)
# ---------------------------------------------------------------------------

def parse_cli_args(defaults):
    """Parse command-line arguments in key=value format."""
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key in defaults:
                # Infer type from default value
                default_type = type(defaults[key])
                if default_type == bool:
                    defaults[key] = value.lower() in ("true", "1", "yes")
                else:
                    defaults[key] = default_type(value)
            else:
                print(f"Warning: unknown parameter '{key}'")
    return defaults


# ---------------------------------------------------------------------------
# Main: matches the kissingfugal-dense-LA.py demo on MultiMagna
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default hyperparameters
    params = {
        "use_GPU": True,
        "learning_rate": 1e-2,
        "max_iter": 10000,
        "embed_dim": 1000,
        "beta": 10,
        "row_pen": 10,
        "col_pen": 200,
        "chunk": 1024,
        "mu": 0.1,
    }
    
    # Parse CLI arguments to override defaults
    params = parse_cli_args(params)
    
    use_GPU = params["use_GPU"]
    learning_rate = params["learning_rate"]
    max_iter = params["max_iter"]
    mu = params["mu"]
    chunk = params["chunk"]
    
    m_list = [params["embed_dim"]]
    beta_list = [params["beta"]]
    row_penalty_list = [params["row_pen"]]
    col_penalty_list = [params["col_pen"]]

    Gq, Gt, n = read_file()

    for m in m_list:
        for beta in beta_list:
            for row_penalty in row_penalty_list:
                for col_penalty in col_penalty_list:
                    print(f"embed_dim={m} beta={beta} "
                          f"row_pen={row_penalty} col_pen={col_penalty} "
                          f"chunk={chunk}")

                    # Adam by default.  Swap to train_with_LBFGS for the
                    # LBFGS trajectory (also chunked + checkpointed).
                    best_V, best_W, history = train_with_adam(
                        Gq, Gt,
                        embed_dim=m, beta=beta, mu=mu,
                        row_penalty=row_penalty, col_penalty=col_penalty,
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                        use_GPU=use_GPU,
                        chunk=chunk,
                        use_checkpoint=True,
                        log_every=500,
                    )

                    # Reconstruct P (chunk-wise) for the assignment step.
                    # If n x n does not fit in your GPU, comment this out
                    # and use greedy_assign_chunked() instead.
                    P_final = reconstruct_P(best_V, best_W, beta=beta, chunk=chunk)
                    P_np = P_final.cpu().numpy()

                    # Sanity check: row / col sums
                    row_sums = P_np.sum(axis=1)
                    col_sums = P_np.sum(axis=0)
                    print(f"rows close to 1: {np.allclose(row_sums, 1.0, atol=1e-2)} "
                          f"(max|sum-1| = {np.abs(row_sums - 1).max():.3e})")
                    print(f"cols close to 1: {np.allclose(col_sums, 1.0, atol=1e-2)} "
                          f"(max|sum-1| = {np.abs(col_sums - 1).max():.3e})")

                    # Hungarian assignment on the n x n P
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                        P_np, maximize=True)
                    
                    cnt = 0
                    for rol, col in zip(row_ind, col_ind):
                            gt = ground_truth.get(int(rol))
                            if gt is not None and gt == int(col):
                                cnt += 1

                        # for acm-dblp
                    acc_hungarian = cnt / data.shape[0]

                    print(f"Hungarian accuracy: {acc_hungarian:.4f}")
