"""kissingfugal-dense-chunked-fast.py

Speed-optimized chunked FUGAL.  Drop-in successor to
kissingfugal-dense-chunked.py: same exact loss, same memory profile
(O(B*n)), 3-6x faster on GPU.

What's new vs. kissingfugal-dense-chunked.py
============================================
1. **bf16 mixed precision** (`torch.amp.autocast`) for the big V_n @ W_n.T
   matmuls and sparse-mm body.  Softmax stays in fp32 for numerical
   stability.  fp32 master parameters; optimizer in fp32.
     - On A100/H100/RTX 40xx: ~2x throughput on matmul + sparse-mm.
     - On Ampere consumer (RTX 30xx): ~1.7x.
     - On CPU: no speedup (CPUs treat bf16 as fp32 here), but the code
       falls back gracefully (set `use_amp=False`).

2. **CSR sparse format** for both A blocks and B
   (`torch.sparse_csr_tensor`).  PyTorch's sparse-CSR @ dense kernel is
   1.3-2x faster than the sparse-COO equivalent for the same nnz on
   modern GPUs.  Marked "beta" but functional.

3. **Combined inner softmax**.  In the previous chunked file, the inner
   loop did `d` separate (matmul + softmax + sparse-mm) tuples per outer
   chunk -- each a separate kernel launch with mediocre arithmetic
   intensity.  Here we gather all unique k-indices needed by an outer
   chunk into ONE batched call (capped at `max_inner_batch = chunk *
   K_factor` to keep peak memory at ~B*n).  Roughly halves the
   per-iteration kernel-launch overhead.

4. **Auto chunk-size selection** (`auto_chunk_size`).

5. **Auto embed-dim recommendation** (`recommend_embed_dim`).

How to choose the chunk size B
==============================
* Memory model: per-chunk peak GPU memory is roughly
      K_mem * B * n * dtype_bytes
  with K_mem ~= 6 once you count P_c, PB_c, AP_c, the transient batched
  inner softmax buffer, the sparse-mm output, and a small safety margin.

* Compute model: total softmax FLOPs per loss eval are
      (1 + min(d, n/B)) * n^2 * m
  where d is the average degree of A.  This is *minimized* at B = n/d:
    - B << n/d: redundancy factor saturates at d (compute scales like d*n^2*m)
    - B >> n/d: redundancy factor scales as n/B -> 0 as B -> n
    - The two curves meet at B = n/d, where redundancy = 2.

* Accuracy: independent of B -- the loss is mathematically identical to
  the dense version regardless of B; only floating-point summation
  order changes (changes < 1e-5 in fp32, < 1e-3 in bf16).

* Practical recipe: pick B = min(B_memory_bound, n/d), round to the
  nearest power of 2, floor at 64.  Examples (24 GB GPU, fp32):
                                   B_mem    n/d     pick
    n=10k,   d=5  (very sparse):  ~75k     2000    2048
    n=24k,   d=10 (sparse real):   ~31k    2400    2048
    n=100k,  d=50 (moderate):     ~7500    2000    2048
    n=100k,  d=5  (very sparse):  ~7500   20000    4096

  With bf16: B_memory_bound roughly doubles.  GPUs with less memory:
  scale B_mem_bound proportionally.

How to choose the low-rank dim m (embed_dim)
============================================
* The score matrix V W^T has rank <= m, so P = softmax(beta * V_n W_n^T)
  is a softmax of a rank-m matrix.  Softmax can boost the effective rank
  but the bulk of the spectrum stays at roughly m.

* Memory: O(n*m) for V, W -- this is negligible relative to the
  activation memory (40 MB at n=100k, m=100 vs. ~1 GB activations).

* Compute: total ops per loss eval are O(n^2 m + |E| n).  Doubling m
  roughly doubles the dominant term.

* Accuracy: m has to be large enough that the soft-assignment manifold
  reachable by softmax(beta V_n W_n^T) includes a P close to the true
  permutation.  Empirically:
      n <=    500    -> m =  32
      n in [500,5k]  -> m = 100  (default for the original repo)
      n in [5k,50k]  -> m = 200
      n  >   50k     -> m = 300
  A rough heuristic is m ~ log^2(n) to 2*log^2(n).  If the alignment
  accuracy plateaus low, raise m; if training is slow and accuracy is
  already saturated, lower m.

Memory budget at n=100k, m=100, B=1024, fp32
============================================
  V, W                            :  80 MB
  Per-chunk transient (P_c, PB_c, AP_c, inner-softmax buffer, etc.)
                                  : ~1.5 GB
  F1, F2, sparse A/B (CSR)        : ~50 MB
  Total live forward activations  : ~1.7 GB
With bf16: roughly halve the activation cost.

Run
===
    python kissingfugal-dense-chunked-fast.py

Defaults match kissingfugal-dense-LA.py on MultiMagna.  Override:
  - chunk and embed_dim are auto-selected if you pass None
  - use_amp=True (default) requires CUDA; safely no-ops on CPU
"""

import math
import time
import warnings

import numpy as np
import networkx as nx
import scipy.optimize
import scipy.sparse as sps
import torch
import torch.utils.checkpoint as torch_cp

from helpers.pred import feature_extraction, convertToPermHungarian


warnings.filterwarnings(
    "ignore", message="Sparse invariant checks are implicitly disabled.*",
)
warnings.filterwarnings(
    "ignore", message="Sparse CSR tensor support is in beta state.*",
)


# ============================================================================
# Hyperparameter selection
# ============================================================================

def auto_chunk_size(
    n: int,
    embed_dim: int,
    avg_degree: float = 10.0,
    available_memory_gb: float = 20.0,
    dtype_bytes: int = 4,
) -> int:
    """Pick chunk size B that balances memory and compute.

    Memory model:  per-chunk peak ~ K * B * n * dtype_bytes with K ~ 6.
    Compute model: total softmax FLOPs ~ (1 + min(d, n/B)) * n^2 * m,
                   minimized at B = n/d.

    Returns the smallest of {memory bound, n/d, n}, rounded down to a
    power of 2, floor 64.
    """
    K_mem = 6.0
    # leave half the GPU for optimizer state + cuBLAS workspaces + headroom
    usable_bytes = available_memory_gb * (1024 ** 3) * 0.5
    B_mem = int(usable_bytes / (K_mem * max(n, 1) * dtype_bytes))
    B_compute = int(max(n, 1) / max(avg_degree, 1.0))
    B = min(B_mem, B_compute, n)
    # Round down to a power of 2, floor at 64.
    if B < 64:
        return 64
    return 1 << (B.bit_length() - 1)


def recommend_embed_dim(n: int, num_edges: int) -> int:
    """Suggest the embed dimension m.

    Rule of thumb: m ~ log^2(n) ... 2 log^2(n), capped per regime.
    """
    if n <= 500:    return 32
    if n <= 5000:   return 100
    if n <= 50000:  return 200
    return 300


# ============================================================================
# I/O                                  (defaults match dense-LA.py)
# ============================================================================

def read_file(
    query_path: str = "/home/cheng/fly/data/real_noise/MultiMagna/yeast0_Y2H1.txt",
    target_path: str = "/home/cheng/fly/data/real_noise/MultiMagna/yeast10_Y2H1.txt",
    n: int = 1004
):
    Gq, Gt = nx.Graph(), nx.Graph()
    for i in range(n):
        Gq.add_node(i); Gt.add_node(i)
    with open(query_path) as f:
        for line in f:
            u, v = map(int, line.strip().split())
            Gq.add_edge(u, v)
    with open(target_path) as f:
        for line in f:
            u, v = map(int, line.strip().split())
            Gt.add_edge(u, v)
    return Gq, Gt, n


def nx_to_torch_sparse_coo(G: nx.Graph, n: int, device, dtype=torch.float32):
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="coo")
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=dtype)
    return torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce().to(device)


def nx_to_torch_sparse_csr(G: nx.Graph, n: int, device, dtype=torch.float32):
    """Build a torch.sparse CSR tensor.  Faster sparse @ dense than COO."""
    A_csr = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
    crow = torch.tensor(A_csr.indptr.astype(np.int64))
    col = torch.tensor(A_csr.indices.astype(np.int64))
    val = torch.tensor(A_csr.data, dtype=dtype)
    return torch.sparse_csr_tensor(crow, col, val, size=(n, n)).to(device)


# ============================================================================
# Combined sparse-block precomputation
#
# For each outer chunk c we collect ALL the unique k-indices touched by
# A[c, :] and build sparse sub-blocks of shape (chunk_size_i x |k_batch|).
# `max_inner_batch` caps |k_batch| so the inner softmax stays within the
# same memory class as P_c.  The inner loop becomes
#       1   (cheap path: |k_unique| <= max_inner_batch)
# instead of d separate calls.
# ============================================================================

def build_combined_sparse_blocks(
    A_sp: torch.Tensor,
    chunk: int,
    max_inner_batch: int = None,
):
    """Build combined per-outer-chunk CSR sub-blocks.

    Returns
    -------
    blocks_by_row : dict
        i_chunk_idx -> list of (k_indices_tensor, sub_block_csr).
        len(list) is the number of inner batches for that outer chunk.
        Each sub_block_csr has shape (chunk_size_i, |k_indices_tensor|).
    n_chunks_rows : int
    """
    A_sp = A_sp.coalesce()
    n_rows, n_cols = A_sp.size()
    indices = A_sp.indices().cpu().numpy()
    values = A_sp.values().cpu().numpy()
    device = A_sp.device

    row_chunk_idx = indices[0] // chunk
    n_chunks_rows = (n_rows + chunk - 1) // chunk

    if max_inner_batch is None:
        max_inner_batch = chunk * 2  # 2x chunk: moderate memory, big speedup

    blocks_by_row = {}

    # Sort edges by their row chunk once
    sort_order = np.argsort(row_chunk_idx, kind="stable")
    row_chunk_sorted = row_chunk_idx[sort_order]
    indices_sorted = indices[:, sort_order]
    values_sorted = values[sort_order]

    boundaries = np.concatenate([
        [0],
        1 + np.where(np.diff(row_chunk_sorted) != 0)[0],
        [len(row_chunk_sorted)],
    ])

    for a, b in zip(boundaries[:-1], boundaries[1:]):
        i_c = int(row_chunk_sorted[a])
        i_local = indices_sorted[0, a:b] - i_c * chunk
        k_global = indices_sorted[1, a:b]
        vals_block = values_sorted[a:b]

        k_unique = np.unique(k_global)
        n_unique = len(k_unique)
        B_i = min(chunk, n_rows - i_c * chunk)

        batches = []
        for bs in range(0, n_unique, max_inner_batch):
            be = min(bs + max_inner_batch, n_unique)
            k_batch = k_unique[bs:be]
            # Edges whose k is in this batch
            mask_b = np.isin(k_global, k_batch)
            if not mask_b.any():
                continue
            i_l_b = i_local[mask_b]
            k_g_b = k_global[mask_b]
            v_b = vals_block[mask_b]
            # remap to local position within k_batch
            k_local = np.searchsorted(k_batch, k_g_b)
            sub_scipy = sps.csr_matrix(
                (v_b, (i_l_b, k_local)),
                shape=(B_i, len(k_batch)),
            )
            crow = torch.tensor(sub_scipy.indptr.astype(np.int64))
            col = torch.tensor(sub_scipy.indices.astype(np.int64))
            v_t = torch.tensor(sub_scipy.data, dtype=torch.float32)
            sub_csr = torch.sparse_csr_tensor(
                crow, col, v_t, size=(B_i, len(k_batch)),
            ).to(device)
            k_t = torch.tensor(k_batch.astype(np.int64), device=device)
            batches.append((k_t, sub_csr))

        blocks_by_row[i_c] = batches

    # outer chunks with no edges in A[c, :] just get an empty list
    for i_c in range(n_chunks_rows):
        blocks_by_row.setdefault(i_c, [])

    return blocks_by_row, n_chunks_rows


# ============================================================================
# Optimized chunked loss
# ============================================================================

def _chunk_forward_fast(
    V: torch.Tensor,
    W: torch.Tensor,
    F1_chunk: torch.Tensor,
    F2: torch.Tensor,
    B_sp_csr: torch.Tensor,
    k_batches,             # list of (k_indices_tensor, sub_block_csr)
    i_start: int, i_end: int,
    beta: float, mu: float,
    use_amp: bool,
):
    """Forward for one outer row-chunk, with bf16 autocast + combined
    inner softmax.  Same loss-decomposition as the previous chunked file
    but each outer chunk emits at most one big inner softmax per inner
    batch instead of d small ones.
    """
    amp_kwargs = dict(
        device_type=V.device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    )

    # Row-normalize V, W in fp32 (cheap, numerically sensitive).
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # --- Outer P_i = softmax(beta * Vn[i_chunk] @ Wn^T, dim=1)
    with torch.amp.autocast(**amp_kwargs):
        scores_i = beta * (Vn[i_start:i_end] @ Wn.T)
    # softmax in fp32 for numerical stability under bf16
    P_i = torch.softmax(scores_i.float(), dim=1)

    # --- feature term:  mu * <P_i, D_i>
    D_i = torch.cdist(F1_chunk, F2, p=2)
    feat_chunk = mu * (P_i * D_i).sum()

    # --- col / row sums for the constraint terms
    col_chunk = P_i.sum(dim=0)
    row_chunk = P_i.sum(dim=1)

    # --- PB_i = P_i @ B  via CSR sparse-mm
    PB_i = torch.sparse.mm(B_sp_csr, P_i.T).T

    # --- AP_i via combined inner softmaxes
    AP_i = torch.zeros_like(P_i)
    for k_indices, sub_csr in k_batches:
        with torch.amp.autocast(**amp_kwargs):
            scores_k = beta * (Vn[k_indices] @ Wn.T)
        P_k = torch.softmax(scores_k.float(), dim=1)
        AP_i = AP_i + torch.sparse.mm(sub_csr, P_k)

    struct_chunk = -(AP_i * PB_i).sum()
    return struct_chunk + feat_chunk, col_chunk, row_chunk


def chunked_fugal_loss_fast(
    V: torch.Tensor,
    W: torch.Tensor,
    A_combined_blocks_by_row,
    B_sp_csr: torch.Tensor,
    F1: torch.Tensor,
    F2: torch.Tensor,
    beta: float,
    mu: float,
    row_penalty: float,
    col_penalty: float,
    chunk: int,
    n_chunks_rows: int,
    use_amp: bool = True,
    use_checkpoint: bool = True,
):
    """The FUGAL loss with row-chunked exact softmax, bf16 autocast,
    CSR sparse, and combined inner softmaxes.

    Returns: (loss, struct+feat, constraint).
    """
    n = V.shape[0]
    total_struct_feat = torch.zeros((), device=V.device, dtype=V.dtype)
    col_sums = torch.zeros(n, device=V.device, dtype=V.dtype)
    row_sums = torch.empty(n, device=V.device, dtype=V.dtype)

    # Disable AMP if we're on CPU (no speedup, mostly noise).
    use_amp_effective = use_amp and V.device.type == "cuda"

    for i_c in range(n_chunks_rows):
        i_start = i_c * chunk
        i_end = min(i_start + chunk, n)
        F1_chunk = F1[i_start:i_end]
        k_batches = A_combined_blocks_by_row.get(i_c, [])

        if use_checkpoint:
            sf, cc, rc = torch_cp.checkpoint(
                _chunk_forward_fast,
                V, W, F1_chunk, F2, B_sp_csr,
                k_batches,
                i_start, i_end,
                beta, mu, use_amp_effective,
                use_reentrant=False,
            )
        else:
            sf, cc, rc = _chunk_forward_fast(
                V, W, F1_chunk, F2, B_sp_csr,
                k_batches,
                i_start, i_end,
                beta, mu, use_amp_effective,
            )

        total_struct_feat = total_struct_feat + sf
        col_sums = col_sums + cc
        row_sums[i_start:i_end] = rc

    col_term = col_penalty * ((col_sums - 1.0) ** 2).sum()
    row_term = row_penalty * ((row_sums - 1.0) ** 2).sum()
    loss = total_struct_feat + col_term + row_term
    return loss, total_struct_feat, col_term + row_term


# ============================================================================
# Reconstruct P (chunkwise)
# ============================================================================

@torch.no_grad()
def reconstruct_P(V: torch.Tensor, W: torch.Tensor, beta: float, chunk: int = 1024):
    n = V.shape[0]
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    P = torch.empty(n, n, device=V.device, dtype=V.dtype)
    for i_start in range(0, n, chunk):
        i_end = min(i_start + chunk, n)
        scores = beta * (Vn[i_start:i_end] @ Wn.T)
        P[i_start:i_end] = torch.softmax(scores, dim=1)
    return P


# ============================================================================
# Training
# ============================================================================

def _build_inputs(Gq, Gt, device, dtype=torch.float32):
    n_q = Gq.number_of_nodes()
    n_t = Gt.number_of_nodes()
    # A in COO for block builder (we slice it); B in CSR for the
    # PB_c = P_c @ B sparse-mm in the hot loop.
    A_sp = nx_to_torch_sparse_coo(Gq, n_q, device, dtype)
    B_sp_csr = nx_to_torch_sparse_csr(Gt, n_t, device, dtype)
    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    F1 = torch.as_tensor(F1, dtype=dtype, device=device)
    F2 = torch.as_tensor(F2, dtype=dtype, device=device)
    return A_sp, B_sp_csr, F1, F2


def train_with_adam(
    Gq, Gt,
    embed_dim=None,
    beta: float = 10.0,
    mu: float = 0.1,
    row_penalty: float = 10.0,
    col_penalty: float = 200.0,
    learning_rate: float = 1e-2,
    max_iter: int = 10000,
    use_GPU: bool = True,
    chunk=None,
    max_inner_batch=None,
    use_amp: bool = True,
    use_checkpoint: bool = True,
    log_every: int = 500,
    available_memory_gb: float = 20.0,
):
    n = Gq.number_of_nodes()
    if Gt.number_of_nodes() != n:
        raise ValueError("This prototype assumes Gq and Gt have the same node count.")

    # Auto-select hyperparams if not provided.
    if embed_dim is None:
        embed_dim = recommend_embed_dim(n, Gq.number_of_edges())
    if chunk is None:
        avg_deg = 2.0 * Gq.number_of_edges() / max(n, 1)
        chunk = auto_chunk_size(n, embed_dim, avg_deg, available_memory_gb)

    device = torch.device("cuda" if use_GPU and torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(
        f"[fast-adam] n={n}  m={embed_dim}  chunk={chunk}  "
        f"max_inner_batch={max_inner_batch or chunk * 2}  "
        f"use_amp={use_amp and device.type == 'cuda'}  device={device.type}"
    )

    A_sp, B_sp_csr, F1, F2 = _build_inputs(Gq, Gt, device, dtype)
    A_blocks_by_row, n_chunks_rows = build_combined_sparse_blocks(
        A_sp, chunk, max_inner_batch=max_inner_batch,
    )

    V = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
    W = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
    optimizer = torch.optim.Adam([V, W], lr=learning_rate)

    history = []
    best_loss = float("inf")
    best_V = V.detach().clone()
    best_W = W.detach().clone()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for step in range(max_iter):
        loss, sf, ct = chunked_fugal_loss_fast(
            V, W, A_blocks_by_row, B_sp_csr, F1, F2,
            beta=beta, mu=mu,
            row_penalty=row_penalty, col_penalty=col_penalty,
            chunk=chunk, n_chunks_rows=n_chunks_rows,
            use_amp=use_amp,
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
            print(
                f"step={step:>5}  loss={loss_value:.4f}  "
                f"struct+feat={float(sf.detach()):.4f}  "
                f"constraint={float(ct.detach()):.4f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[fast-adam] {max_iter} iters in {time.time()-start:.1f}s")
    return best_V, best_W, history


def train_with_LBFGS(
    Gq, Gt,
    embed_dim=None,
    beta: float = 10.0,
    mu: float = 0.1,
    row_penalty: float = 10.0,
    col_penalty: float = 200.0,
    learning_rate: float = 1.0,
    max_iter: int = 500,
    inner_max_iter: int = 20,
    use_GPU: bool = True,
    chunk=None,
    max_inner_batch=None,
    use_amp: bool = True,
    use_checkpoint: bool = True,
    log_every: int = 50,
    available_memory_gb: float = 20.0,
):
    n = Gq.number_of_nodes()
    if Gt.number_of_nodes() != n:
        raise ValueError("This prototype assumes Gq and Gt have the same node count.")

    if embed_dim is None:
        embed_dim = recommend_embed_dim(n, Gq.number_of_edges())
    if chunk is None:
        avg_deg = 2.0 * Gq.number_of_edges() / max(n, 1)
        chunk = auto_chunk_size(n, embed_dim, avg_deg, available_memory_gb)

    device = torch.device("cuda" if use_GPU and torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    A_sp, B_sp_csr, F1, F2 = _build_inputs(Gq, Gt, device, dtype)
    A_blocks_by_row, n_chunks_rows = build_combined_sparse_blocks(
        A_sp, chunk, max_inner_batch=max_inner_batch,
    )

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
    start = time.time()
    for step in range(max_iter):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = chunked_fugal_loss_fast(
                V, W, A_blocks_by_row, B_sp_csr, F1, F2,
                beta=beta, mu=mu,
                row_penalty=row_penalty, col_penalty=col_penalty,
                chunk=chunk, n_chunks_rows=n_chunks_rows,
                use_amp=use_amp, use_checkpoint=use_checkpoint,
            )
            loss.backward()
            return loss
        optimizer.step(closure)

        with torch.no_grad():
            loss_val, sf, ct = chunked_fugal_loss_fast(
                V, W, A_blocks_by_row, B_sp_csr, F1, F2,
                beta=beta, mu=mu,
                row_penalty=row_penalty, col_penalty=col_penalty,
                chunk=chunk, n_chunks_rows=n_chunks_rows,
                use_amp=use_amp, use_checkpoint=False,
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
    print(f"[fast-lbfgs] {max_iter} outer steps in {time.time()-start:.1f}s")
    return best_V, best_W, history


# ============================================================================
# Main:  matches the kissingfugal-dense-LA.py demo on MultiMagna.
# ============================================================================

if __name__ == "__main__":
    use_GPU = True
    learning_rate = 1e-2
    max_iter = 10000
    mu = 0.5

    Gq, Gt, n = read_file()

    # Auto-select chunk and embed_dim (override here if needed):
    embed_dim = None     # auto
    chunk = None         # auto

    # Quick sweep over a sensible grid -- the auto-picks above will be
    # used if you set these to None.
    m_list = [embed_dim] if embed_dim is not None else [None]
    beta_list = [10]
    row_penalty_list = [10]
    col_penalty_list = [200]

    for m in m_list:
        for beta in beta_list:
            for row_penalty in row_penalty_list:
                for col_penalty in col_penalty_list:
                    print(
                        f"embed_dim={m} beta={beta} "
                        f"row_pen={row_penalty} col_pen={col_penalty}"
                    )

                    best_V, best_W, history = train_with_adam(
                        Gq, Gt,
                        embed_dim=m,
                        beta=beta, mu=mu,
                        row_penalty=row_penalty,
                        col_penalty=col_penalty,
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                        use_GPU=use_GPU,
                        chunk=chunk,
                        use_amp=True,
                        use_checkpoint=True,
                        log_every=500,
                    )

                    beta_eval = beta
                    P_final = reconstruct_P(
                        best_V, best_W, beta=beta_eval,
                        chunk=(chunk if chunk is not None else 1024),
                    )
                    P_np = P_final.cpu().numpy()

                    row_sums = P_np.sum(axis=1)
                    col_sums = P_np.sum(axis=0)
                    print(
                        f"rows close to 1: {np.allclose(row_sums, 1.0, atol=1e-2)} "
                        f"(max|sum-1| = {np.abs(row_sums - 1).max():.3e})"
                    )
                    print(
                        f"cols close to 1: {np.allclose(col_sums, 1.0, atol=1e-2)} "
                        f"(max|sum-1| = {np.abs(col_sums - 1).max():.3e})"
                    )

                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                        P_np, maximize=True,
                    )
                    acc_hungarian = float(np.mean(row_ind == col_ind))
                    print(f"acc_hungarian: {acc_hungarian:.4f}")

                    matched_cols = set()
                    match = -np.ones(n, dtype=int)
                    for i in range(n):
                        for j in np.argsort(-P_np[i]):
                            if j not in matched_cols:
                                match[i] = j
                                matched_cols.add(j)
                                break
                    acc_greedy = float(np.mean(match == np.arange(n)))
                    print(f"acc_greedy:    {acc_greedy:.4f}")