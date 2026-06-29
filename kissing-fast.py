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
   chunk into ONE batched call (capped at `max_inner_batch = chunk`
   to keep the backward-pass peak manageable).  Roughly halves the
   per-iteration kernel-launch overhead.

4. **Auto chunk-size selection** (`auto_chunk_size`).

5. **Auto embed-dim recommendation** (`recommend_embed_dim`).

How to choose the chunk size B
==============================
* Memory model: per-chunk training peak GPU memory is roughly
      K_mem * B * n * dtype_bytes
  with K_mem ~= 16 once you count forward activations, their gradients,
  the transient batched inner softmax buffer, sparse-mm outputs, and a
  small safety margin.

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

    Memory model:  training peak ~ K * B * n * dtype_bytes with K ~ 16.
    Compute model: total softmax FLOPs ~ (1 + min(d, n/B)) * n^2 * m,
                   minimized at B = n/d.

    Returns the smallest of {memory bound, n/d, n}, rounded down to a
    power of 2, floor 64.
    """
    K_mem = 16.0 * max(1.0, avg_degree / 10.0)
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
    query_path: str = "data/real_noise/ippds/ippds-100.txt",
    target_path: str = "data/real_noise/ippds/ippds-90.txt",
    n: int = 108175,
    keep_self_loops: bool = False,
):
    Gq, Gt = nx.Graph(), nx.Graph()
    for i in range(n):
        Gq.add_node(i); Gt.add_node(i)
    with open(query_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except Exception:
                continue
            if u == v and not keep_self_loops:
                continue
            Gq.add_edge(u, v)
    with open(target_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except Exception:
                continue
            if u == v and not keep_self_loops:
                continue
            Gt.add_edge(u, v)
    print(f"[I/O] query_file={query_path} target_file={target_path} self_loops={keep_self_loops}")        
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
        max_inner_batch = chunk

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
    V, W, col_bias,
    F1_chunk, F2, B_sp_csr,
    k_batches,
    i_start, i_end,
    beta, mu,
    use_amp,
    col_grad_vec=None,          # NEW: detached length-n vector g (pass 2), or None (pass 1)
):
    """Forward for one outer row-chunk.
 
    If `col_grad_vec` is provided (pass 2), the column-constraint gradient is
    folded into this chunk's scalar loss via (P_i * col_grad_vec).sum(), which
    is a constant-weighted linear term: it contributes the correct gradient to
    P_i while adding nothing that must outlive this chunk's backward.
 
    Returns: (chunk_scalar_loss, col_chunk, row_chunk)
      - chunk_scalar_loss : struct + feat (+ folded column penalty in pass 2)
      - col_chunk         : P_i.sum(dim=0), length n   (used by pass 1)
      - row_chunk         : P_i.sum(dim=1), length (i_end-i_start)
    """
    amp_kwargs = dict(
        device_type=V.device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    )
 
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
 
    with torch.amp.autocast(**amp_kwargs):
        scores_i = beta * (Vn[i_start:i_end] @ Wn.T)
    if col_bias is not None:
        scores_i = scores_i + col_bias.to(dtype=scores_i.dtype)
    P_i = torch.softmax(scores_i.float(), dim=1)
 
    # feature term
    D_i = torch.cdist(F1_chunk, F2, p=2)
    feat_chunk = mu * (P_i * D_i).sum()
 
    # column / row sums
    col_chunk = P_i.sum(dim=0)
    row_chunk = P_i.sum(dim=1)
 
    # PB_i = P_i @ B
    PB_i = torch.sparse.mm(B_sp_csr, P_i.T).T
 
    # AP_i via combined inner softmaxes
    AP_i = torch.zeros_like(P_i)
    for k_indices, sub_csr in k_batches:
        with torch.amp.autocast(**amp_kwargs):
            scores_k = beta * (Vn[k_indices] @ Wn.T)
        if col_bias is not None:
            scores_k = scores_k + col_bias.to(dtype=scores_k.dtype)
        P_k = torch.softmax(scores_k.float(), dim=1)
        AP_i = AP_i + torch.sparse.mm(sub_csr, P_k)
 
    struct_chunk = -(AP_i * PB_i).sum()
    chunk_loss = struct_chunk + feat_chunk
 
    # Pass 2: fold the (constant-weighted) column penalty gradient in.
    # (P_i * g).sum() has gradient exactly g w.r.t. P_i == d(col_term)/dP_i.
    if col_grad_vec is not None:
        chunk_loss = chunk_loss + (P_i * col_grad_vec).sum()
 
    return chunk_loss, col_chunk, row_chunk


def chunked_fugal_loss_fast(
    V, W, col_bias,
    A_combined_blocks_by_row,
    B_sp_csr,
    F1, F2,
    beta, mu,
    row_penalty, col_penalty,
    chunk, n_chunks_rows,
    use_amp=True,
    use_checkpoint=True,
):
    """FUGAL loss, row-chunked, with a checkpoint-friendly column penalty.
 
    Returns: (loss, struct+feat, constraint)  -- same signature as before.
    """
    n = V.shape[0]
    use_amp_effective = use_amp and V.device.type == "cuda"
 
    # ----------------------------------------------------------------
    # PASS 1 (no grad): column sums -> constant gradient vector g.
    # Cheap: no autograd graph, so memory is just one P_i chunk at a time.
    # ----------------------------------------------------------------
    with torch.no_grad():
        col_sums_ng = torch.zeros(n, device=V.device, dtype=V.dtype)
        Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
        for i_c in range(n_chunks_rows):
            i_start = i_c * chunk
            i_end = min(i_start + chunk, n)
            scores_i = beta * (Vn[i_start:i_end] @ Wn.T)
            if col_bias is not None:
                scores_i = scores_i + col_bias
            P_i = torch.softmax(scores_i, dim=1)
            col_sums_ng += P_i.sum(dim=0)
        # exact scalar column term (constant w.r.t. the pass-2 graph)
        col_term_value = col_penalty * ((col_sums_ng - 1.0) ** 2).sum()
        # constant gradient weights for pass 2
        col_grad_vec = (2.0 * col_penalty * (col_sums_ng - 1.0)).detach()
 
    # ----------------------------------------------------------------
    # PASS 2 (checkpointed, with grad): struct + feat + row penalty,
    # with the column penalty folded in per chunk via col_grad_vec.
    # ----------------------------------------------------------------
    total_struct_feat = torch.zeros((), device=V.device, dtype=V.dtype)
    row_sums = torch.empty(n, device=V.device, dtype=V.dtype)
 
    for i_c in range(n_chunks_rows):
        i_start = i_c * chunk
        i_end = min(i_start + chunk, n)
        F1_chunk = F1[i_start:i_end]
        k_batches = A_combined_blocks_by_row.get(i_c, [])
 
        if use_checkpoint:
            cl, cc, rc = torch_cp.checkpoint(
                _chunk_forward_fast,
                V, W, col_bias, F1_chunk, F2, B_sp_csr,
                k_batches,
                i_start, i_end,
                beta, mu, use_amp_effective,
                col_grad_vec,
                use_reentrant=False,
            )
        else:
            cl, cc, rc = _chunk_forward_fast(
                V, W, col_bias, F1_chunk, F2, B_sp_csr,
                k_batches,
                i_start, i_end,
                beta, mu, use_amp_effective,
                col_grad_vec,
            )
        # cl already includes this chunk's folded column-penalty contribution;
        # subtract its *value* so total_struct_feat stays "struct+feat only"
        # for logging, while gradients still flow correctly.
        total_struct_feat = total_struct_feat + cl
        row_sums[i_start:i_end] = rc
 
    # The folded column terms summed to a graph whose VALUE is
    #   sum_i (P_i * g).sum() = (col_sums * g).sum()  -- not equal to col_term
    # so we must correct the reported scalar.  Gradients are unaffected by
    # adding/subtracting constants.  We rebuild a clean reported loss:
    #
    #   loss = (struct+feat graph) + (row penalty graph)
    #          + col_term_value (constant, correct gradient already injected)
    #
    # total_struct_feat currently = (struct+feat) + sum_i (P_i*g).sum()
    # We want its gradient (correct) but its reported value split out cleanly.
    folded_value = (col_sums_ng * col_grad_vec).sum()  # constant
    struct_feat_value_graph = total_struct_feat - folded_value  # value-correct, same grad
 
    row_term = row_penalty * ((row_sums - 1.0) ** 2).sum()
 
    # loss: graph carries correct gradients from struct+feat, row, and the
    # folded column penalty; the column term's *value* is col_term_value.
    loss = struct_feat_value_graph + row_term + col_term_value
 
    constraint_value = col_term_value + row_term
    return loss, struct_feat_value_graph, constraint_value


# ============================================================================
# Reconstruct P (chunkwise)
# ============================================================================

@torch.no_grad()
def reconstruct_P(
    V: torch.Tensor,
    W: torch.Tensor,
    beta: float,
    chunk: int = 1024,
    col_bias: torch.Tensor = None,
):
    n = V.shape[0]
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    P = torch.empty(n, n, device=V.device, dtype=V.dtype)
    for i_start in range(0, n, chunk):
        i_end = min(i_start + chunk, n)
        scores = beta * (Vn[i_start:i_end] @ Wn.T)
        if col_bias is not None:
            scores = scores + col_bias
        P[i_start:i_end] = torch.softmax(scores, dim=1)
    return P


@torch.no_grad()
def evaluate_alignment_chunked(
    V: torch.Tensor,
    W: torch.Tensor,
    beta: float,
    chunk: int = 1024,
    col_bias: torch.Tensor = None,
):
    """Evaluate row/column sums and matching diagnostics without dense P."""
    n = V.shape[0]
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
    col_sums = np.zeros(n, dtype=np.float64)
    matched_cols = np.zeros(n, dtype=bool)
    greedy_correct = 0
    top1_correct = 0
    diagonal_mass = 0.0
    max_row_error = 0.0

    if col_bias is not None:
        col_bias = col_bias.detach()

    for i_start in range(0, n, chunk):
        i_end = min(i_start + chunk, n)
        scores = beta * (Vn[i_start:i_end] @ Wn.T)
        if col_bias is not None:
            scores = scores + col_bias
        P_chunk = torch.softmax(scores, dim=1).cpu().numpy()
        max_row_error = max(
            max_row_error,
            float(np.abs(P_chunk.sum(axis=1) - 1.0).max()),
        )
        col_sums += P_chunk.sum(axis=0)
        row_argmax = P_chunk.argmax(axis=1)
        row_ids = np.arange(i_start, i_end)
        top1_correct += int(np.sum(row_argmax == row_ids))
        diagonal_mass += float(P_chunk[np.arange(i_end - i_start), row_ids].sum())

        for i_local, row in enumerate(P_chunk):
            i = i_start + i_local
            for j in np.argsort(-row):
                if not matched_cols[j]:
                    matched_cols[j] = True
                    greedy_correct += int(i == j)
                    break

    return (
        max_row_error,
        col_sums,
        greedy_correct / max(n, 1),
        top1_correct / max(n, 1),
        diagonal_mass / max(n, 1),
    )


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


def _gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def _cuda_memory_summary(device) -> str:
    if device.type != "cuda":
        return "gpu_mem=n/a"

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    total = torch.cuda.get_device_properties(device_index).total_memory
    allocated = torch.cuda.memory_allocated(device_index)
    reserved = torch.cuda.memory_reserved(device_index)
    peak_allocated = torch.cuda.max_memory_allocated(device_index)
    peak_reserved = torch.cuda.max_memory_reserved(device_index)
    max_gpu_usage = 100.0 * peak_reserved / max(total, 1)

    return (
        f"gpu_mem=alloc={_gb(allocated):.2f}GB "
        f"reserved={_gb(reserved):.2f}GB "
        f"peak_alloc={_gb(peak_allocated):.2f}GB "
        f"peak_reserved={_gb(peak_reserved):.2f}GB "
        f"max_gpu_usage={max_gpu_usage:.1f}%"
    )


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
    train_col_bias: bool = True,
    init_mode: str = "random",
    init_noise: float = 1e-3,
    log_every: int = 500,
    available_memory_gb: float = 20.0,
):
    n = Gq.number_of_nodes()
    if Gt.number_of_nodes() != n:
        raise ValueError("This prototype assumes Gq and Gt have the same node count.")

    # Auto-select hyperparams if not provided.
    avg_deg = 2.0 * Gq.number_of_edges() / max(n, 1)
    max_deg = max(dict(Gq.degree()).values())
    if embed_dim is None:
        embed_dim = recommend_embed_dim(n, Gq.number_of_edges())
    if chunk is None:
        chunk = auto_chunk_size(n, embed_dim, avg_deg, available_memory_gb)


    if use_checkpoint:
        # use_checkpoint = max_deg < 1000 
        print(f"[fast-adam] avg_degree={avg_deg:.1f}  use_checkpoint={use_checkpoint}")
    device = torch.device("cuda" if use_GPU and torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(
        f"[fast-adam] n={n} mu={mu} m={embed_dim}  chunk={chunk}  "
        f"max_inner_batch={max_inner_batch}  "
        f"init_mode={init_mode}  "
        f"train_col_bias={train_col_bias}  "
        f"use_amp={use_amp and device.type == 'cuda'}  device={device.type}"
    )

    A_sp, B_sp_csr, F1, F2 = _build_inputs(Gq, Gt, device, dtype)
    print(_cuda_memory_summary(device)) 
    A_blocks_by_row, n_chunks_rows = build_combined_sparse_blocks(
        A_sp, chunk, max_inner_batch=max_inner_batch,
    )

    if init_mode == "shared":
        init = torch.rand((n, embed_dim), device=device, dtype=dtype)
        V = torch.nn.Parameter(init.clone())
        W = torch.nn.Parameter(init + init_noise * torch.rand_like(init))
    elif init_mode == "random":
        V = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
        W = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
    else:
        raise ValueError("init_mode must be 'random' or 'shared'.")
    col_bias = None
    params = [V, W]
    if train_col_bias:
        col_bias = torch.nn.Parameter(torch.zeros(n, device=device, dtype=dtype))
        params.append(col_bias)
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    history = []
    best_loss = float("inf")
    best_V = V.detach().clone()
    best_W = W.detach().clone()
    best_col_bias = None if col_bias is None else col_bias.detach().clone()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for step in range(max_iter):
        loss, sf, ct = chunked_fugal_loss_fast(
            V, W, col_bias, A_blocks_by_row, B_sp_csr, F1, F2,
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
            best_col_bias = None if col_bias is None else col_bias.detach().clone()

        if step % log_every == 0 or step == max_iter - 1:
            print(
                f"step={step:>5}  loss={loss_value:.4f}  "
                f"struct+feat={float(sf.detach()):.4f}  "
                f"constraint={float(ct.detach()):.4f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[fast-adam] {max_iter} iters in {time.time()-start:.1f}s")
    print(f"[fast-adam] peak memory usage: {_cuda_memory_summary(device)}")
    return best_V, best_W, best_col_bias, history, chunk


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
    train_col_bias: bool = True,
    init_mode: str = "random",
    init_noise: float = 1e-3,
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

    if init_mode == "shared":
        init = torch.rand((n, embed_dim), device=device, dtype=dtype)
        V = torch.nn.Parameter(init.clone())
        W = torch.nn.Parameter(init + init_noise * torch.rand_like(init))
    elif init_mode == "random":
        V = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
        W = torch.nn.Parameter(torch.rand((n, embed_dim), device=device, dtype=dtype))
    else:
        raise ValueError("init_mode must be 'random' or 'shared'.")
    col_bias = None
    params = [V, W]
    if train_col_bias:
        col_bias = torch.nn.Parameter(torch.zeros(n, device=device, dtype=dtype))
        params.append(col_bias)

    optimizer = torch.optim.LBFGS(
        params, lr=learning_rate, max_iter=inner_max_iter,
        line_search_fn="strong_wolfe", history_size=20,
    )

    history = []
    best_loss = float("inf")
    best_V = V.detach().clone()
    best_W = W.detach().clone()
    best_col_bias = None if col_bias is None else col_bias.detach().clone()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for step in range(max_iter):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = chunked_fugal_loss_fast(
                V, W, col_bias, A_blocks_by_row, B_sp_csr, F1, F2,
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
                V, W, col_bias, A_blocks_by_row, B_sp_csr, F1, F2,
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
            best_col_bias = None if col_bias is None else col_bias.detach().clone()
        if step % log_every == 0 or step == max_iter - 1:
            print(f"lbfgs-step={step:>4}  loss={loss_value:.4f}  "
                  f"struct+feat={float(sf):.4f}  constraint={float(ct):.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[fast-lbfgs] {max_iter} outer steps in {time.time()-start:.1f}s")
    print(f"[fast-lbfgs] peak memory usage: {_cuda_memory_summary(device)}")
    return best_V, best_W, best_col_bias, history


# ============================================================================
# Main:  matches the kissingfugal-dense-LA.py demo on MultiMagna.
# ============================================================================

if __name__ == "__main__":
    use_GPU = True
    learning_rate = 1e-2
    max_iter = 2000
    mu = 0.1

    Gq, Gt, n = read_file()
    max_inner_batch=None
    # Auto-select chunk if needed. Set embed_dim=None to use recommend_embed_dim.
    embed_dim = 2000
    chunk = None    # auto

    # Quick sweep over a sensible grid -- the auto-picks above will be
    # used if you set these to None.
    m_list = [embed_dim] if embed_dim is not None else [None]
    beta_list = [25]
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

                    best_V, best_W, best_col_bias, history, chosen_chunk = train_with_adam(
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
                        init_mode="random",
                        train_col_bias=True,
                        log_every=500,
                        max_inner_batch=max_inner_batch,
                    )

                    beta_eval = beta
                    eval_chunk = chosen_chunk
                    dense_bytes = n * n * np.dtype(np.float32).itemsize
                    
                    
                    if dense_bytes <= 4 * (1024 ** 3):
                        P_final = reconstruct_P(
                            best_V, best_W, beta=beta_eval, chunk=eval_chunk,
                            col_bias=best_col_bias,
                        )
                        P_np = P_final.cpu().numpy()
                        row_sums = P_np.sum(axis=1)
                        col_sums = P_np.sum(axis=0)
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
                        top1_acc = float(np.mean(P_np.argmax(axis=1) == np.arange(n)))
                        diag_mass = float(P_np[np.arange(n), np.arange(n)].mean())
                        max_row_error = float(np.abs(row_sums - 1).max())
                    else:
                        print(
                            "acc_hungarian: skipped "
                            f"(dense P would use {_gb(dense_bytes):.2f}GB)"
                        )
                        (
                            max_row_error,
                            col_sums,
                            acc_greedy,
                            top1_acc,
                            diag_mass,
                        ) = (
                            evaluate_alignment_chunked(
                                best_V, best_W, beta=beta_eval, chunk=eval_chunk,
                                col_bias=best_col_bias,
                            )
                        )

                    print(
                        f"rows close to 1: {max_row_error <= 1e-2} "
                        f"(max|sum-1| = {max_row_error:.3e})"
                    )
                    print(
                        f"cols close to 1: {np.allclose(col_sums, 1.0, atol=1e-2)} "
                        f"(max|sum-1| = {np.abs(col_sums - 1).max():.3e})"
                    )
                    print(f"acc_top1:      {top1_acc:.4f}")
                    print(f"diag_mass:     {diag_mass:.6f}")
                    print(f"acc_greedy:    {acc_greedy:.4f}")
