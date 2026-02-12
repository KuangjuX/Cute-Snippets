# HTile Design Philosophy

HTile is a hardware-aware abstraction layer built on top of **CuTe** (NVIDIA's C++ Template Library for CUDA Tensor Cores, exposed via Python bindings). Its primary goal is to simplify the development of high-performance CUDA kernels by providing reusable, composable primitives for memory hierarchy traversal and tile management.

This project draws significant inspiration from **[quack](https://github.com/KuangjuX/quack/tree/main)** (A Quirky Assortment of CuTe Kernels), adapting its high-performance patterns into a composable framework, and from **[TileFusion](https://github.com/microsoft/TileFusion)**, whose programming model motivates the separation of tile configuration from dataflow.

---

## Core Philosophy: Separation of Concerns

HTile rigorously separates three orthogonal aspects of kernel development:

| Aspect | What it answers | HTile Component |
|--------|----------------|-----------------|
| **Configuration** | *How many threads? What tile shape? Which cluster size?* | `ElementwiseTileConfig` |
| **Data Movement** | *How does data flow G→S→R and R→G?* | `ElementwiseLoader`, `ElementwiseStorer`, `ElementwiseKernelContext` |
| **Computation** | *What math do we perform on register data?* | Kernel-specific code (e.g., `Softmax.kernel`, `RMSNorm.kernel`) |

This separation allows kernel developers to focus on the algorithm (e.g., "compute RMSNorm") without rewriting the complex boilerplate for efficient memory loading, async copies, and barrier synchronization for every new kernel.

### Motivation: The Duplication Problem

Before HTile, elementwise reduction kernels (Softmax, RMSNorm, LayerNorm, Cross-Entropy, etc.) shared a massive amount of identical infrastructure code:

```
┌─────────────────────────────────────────────────────────────────┐
│  Common Boilerplate (duplicated across every kernel)            │
│  ─────────────────────────────────────────────────────────────  │
│  • Thread/block/cluster index extraction                        │
│  • Shared memory allocation (data tile + reduction buffer)      │
│  • TiledCopy construction and thread-level partitioning         │
│  • Boundary predicate generation (predicate_k)                  │
│  • Register fragment creation                                   │
│  • Mbarrier initialization and cluster synchronization          │
│  • Async copy pipeline (cp_async_commit_group / wait_group)     │
│  • Vectorized copy with predicate                               │
│  • Row-bounds checking                                          │
├─────────────────────────────────────────────────────────────────┤
│  Algorithm-Specific Code (unique per kernel)                    │
│  ─────────────────────────────────────────────────────────────  │
│  • Softmax: max_reduce → exp → sum_reduce → normalize          │
│  • RMSNorm: sum_sq → rsqrt → scale → bias                      │
│  • LayerNorm: mean → variance → normalize → scale → bias       │
└─────────────────────────────────────────────────────────────────┘
```

HTile extracts the common boilerplate into reusable components, so each new kernel only needs to implement the algorithm-specific portion.

---

## Architecture Overview

### TileFusion Concept Mapping

The following table maps TileFusion's programming concepts to their HTile equivalents:

| TileFusion Concept | HTile Equivalent | Location |
|---|---|---|
| GlobalTile shape/layout | `tiler_mn`, `tiled_copy` | `config.py` |
| WarpLayout | `thr_layout` (derived from `threads_per_row`) | `config.py` → `VectorCopy` |
| TileIterator | `cute.local_tile(mT, tiler_mn, coord)` | `dataflow.py` |
| GlobalToRegLoader | `ElementwiseLoader` (G→S→R pipeline) | `dataflow.py` |
| RegToGlobalStorer | `ElementwiseStorer` (R→G pipeline) | `dataflow.py` |
| RegTile | `VectorRegTile` / `make_fragment_like` | `types/register.py` |
| Tile Primitive Config | `ElementwiseTileConfig` | `config.py` |

### 1. `ElementwiseTileConfig` — Tile Primitive Configuration

The `ElementwiseTileConfig` class encapsulates all **static, hardware-aware configuration decisions**. It defines the "shape" of the computation on the hardware without knowing anything about the specific algorithm.

**Responsibilities:**

- **Thread heuristics:** Determines `num_threads` and `threads_per_row` based on the problem size \( N \).
- **Cluster auto-tuning:** Selects the optimal Thread Block Cluster size for Hopper (SM90+) based on \( N \) and `dtype.width`.
- **VectorCopy construction:** Creates `VectorCopy` objects that encapsulate the copy atom, thread layout, and value layout for vectorized memory access.
- **Tiler computation:** Computes `tiler_mn = (tile_M, tile_N)` for `cute.local_tile`.
- **Reduction layout:** Constructs a `ReduceLayout` for the shared memory reduction buffer.
- **Shared memory allocation:** Allocates the reduction buffer and mbarrier arrays.
- **Cluster initialization:** Initializes mbarriers and performs cluster synchronization.

```python
# Example: Configuration for RMSNorm with auto cluster detection
config = ElementwiseTileConfig(
    dtype=cutlass.Float16,
    N=4096,           # Hidden dimension
    cluster_n=None,   # Auto-detect best cluster size
    stage=1,          # 1 reduction stage (sum_sq only)
)

# The config automatically determines:
#   num_threads = 128      (N ≤ 16384)
#   threads_per_row = 32   (N ≤ 3072 → 32)
#   cluster_n = 1          (N ≤ 16k for fp16)
```

**Thread / Warp / Cluster Heuristics:**

| N range | `threads_per_row` | `num_threads` |
|---------|-------------------|---------------|
| ≤ 64 | 8 | 128 |
| ≤ 128 | 16 | 128 |
| ≤ 3072 | 32 | 128 |
| ≤ 6144 | 64 | 128 |
| ≤ 16384 | 128 | 128 |
| > 16384 | 256 | 256 |

**Cluster size heuristics (fp16):**

| N range | `cluster_n` |
|---------|-------------|
| ≤ 16K | 1 |
| ≤ 32K | 2 |
| ≤ 64K | 4 |
| ≤ 128K | 8 |
| > 128K | 16 |

### 2. `ElementwiseKernelContext` — Dataflow Orchestrator

The `ElementwiseKernelContext` is the **runtime orchestrator** that runs inside the `@cute.kernel`. It performs the entire common setup sequence in a single `ctx.setup()` call.

**What `setup()` does (in order):**

1. **Index extraction:** Computes `tidx`, `bidx`, `cluster_y`.
2. **Shared memory allocation:** Allocates `sX` (input tile), optional extra smem tensors (e.g., for residual), reduction buffer, and mbarrier array.
3. **Tile partitioning:** Uses `cute.local_tile` to slice the global tensor, then `tiled_copy.get_slice(tidx)` to partition into thread-local views (`tXgX`, `tXsX`).
4. **Predicate generation:** Computes `is_even_N` and generates `tXpX` via `predicate_k` when needed.
5. **Fragment creation:** Allocates register fragments (`tXrX`).
6. **Cluster initialization:** Initializes mbarriers and synchronizes the cluster.

After `setup()`, the kernel has access to:

| Attribute | Description |
|-----------|-------------|
| `ctx.tXgX` | Partitioned global source tensor |
| `ctx.tXsX` | Partitioned shared memory tensor |
| `ctx.tXrX` | Register fragment for input |
| `ctx.tXcX` | Coordinate tensor for boundary checks |
| `ctx.tXpX` | Predicate tensor (or `None` if evenly tiled) |
| `ctx.reduction_buffer` | Shared memory reduction buffer |
| `ctx.mbar_ptr` | Mbarrier pointer (or `None` if `cluster_n == 1`) |
| `ctx.vector_copy_fn` | `partial(vector_copy, pred=tXpX)` |
| `ctx.row` | Current row index |
| `ctx.row_in_bounds` | Whether the current row is within bounds |
| `ctx.shape` | Shape of the global input tensor |

```python
@cute.kernel
def kernel(self, mX, mO, tiler_mn, tiled_copy, threads_per_row):
    ctx = ElementwiseKernelContext(self.config)
    ctx.setup(mX, tiler_mn, tiled_copy)

    # Everything is ready — focus on the algorithm:
    ctx.load_to_smem_and_regs()          # G → S → R
    x = ctx.tXrX.load().to(cute.Float32) # Use register data
    # ... compute ...
    ctx.store_from_regs(tXrO, tXgO)      # R → G
```

**Extra shared memory tensors:** For kernels that need additional smem buffers (e.g., RMSNorm with residual), pass `extra_smem_tensors=1` to `setup()`. The extra tensors are accessible via `ctx.extra_sX[i]`.

### 3. Dataflow Primitives (`Loader` & `Storer`)

HTile provides standardized pipelines for moving data through the memory hierarchy.

**`ElementwiseLoader` (G → S → R):**

```
Global Memory ──[cp.async]──▶ Shared Memory ──[autovec_copy]──▶ Registers
                  │                                │
           cp_async_commit_group              (synchronous)
           cp_async_wait_group(0)
```

- Issues asynchronous copy instructions (`cp.async`) from Global to Shared memory.
- Manages `cp_async_commit_group` and `wait_group` barriers.
- Loads from Shared Memory to Registers via `autovec_copy`.
- Handles boundary checks via predicate tensors automatically.

**`ElementwiseStorer` (R → G):**

- Stores results from Registers to Global Memory.
- Handles vectorized stores and boundary predicates.
- Respects row-bounds checking.

### 4. Compute Primitives

HTile includes optimized implementations for common reduction patterns.

**`row_reduce`:**

- Performs efficient row-wise reduction (e.g., `ADD`, `MAX`) within a thread block or across a cluster.
- Dispatches to `block_reduce` (single CTA) or `cluster_reduce` (multi-CTA via Distributed Shared Memory) based on `mbar_ptr`.
- Pipeline: thread-local reduce → warp shuffle → cross-warp smem reduce → (optional) cross-CTA cluster reduce.

**`ReduceLayout`:**

The reduction buffer has shape `(num_warps // warps_per_row, (warps_per_row, cluster_n), stage)` with memory order `(1, 0, 2)`:

- **Dimension 0** — Logical rows being processed concurrently.
- **Dimension 1** — `(warps_per_row, cluster_n)`: reduction peers within a row and across cluster.
- **Dimension 2** — Pipeline stages (e.g., stage 0 for MAX, stage 1 for SUM in softmax).

The `(1, 0, 2)` ordering ensures coalesced access for the reduction peer dimension and logical isolation between pipeline stages.

### 5. Copy Primitives

**`VectorCopy`:**

Encapsulates the construction of CuTe `TiledCopy` objects for vectorized memory access:

- Configures copy atom with appropriate `num_bits_per_copy` (up to 128 bits).
- Builds thread layout `(num_threads // threads_per_row, threads_per_row)` with order `(1, 0)`.
- Provides `predicate_k` for boundary handling.
- Supports both synchronous and asynchronous (`cp.async`) copy modes.

---

## The Kernel Lifecycle

A typical kernel written with HTile follows this structure:

```
┌──────────────────────────────────────────────────────────────┐
│  Host Side (@cute.jit __call__)                              │
│  ────────────────────────────────────────────────────────    │
│  1. Compute vec_size, tiler_mn, tiled_copy                   │
│  2. Expand 1D tensors (weight, bias) to match tile shape     │
│  3. Launch kernel with grid/block/cluster dimensions          │
├──────────────────────────────────────────────────────────────┤
│  Device Side (@cute.kernel)                                  │
│  ────────────────────────────────────────────────────────    │
│  4. ctx = ElementwiseKernelContext(config)                    │
│  5. ctx.setup(mX, tiler_mn, tiled_copy)                      │
│     ├── Allocate smem (data + reduction + mbar)              │
│     ├── Partition global/smem tiles                           │
│     ├── Generate predicates                                  │
│     ├── Create register fragments                            │
│     └── Initialize cluster barriers                          │
│  6. Load: G → S → R (ctx.load_to_smem_and_regs)             │
│  7. Compute: Algorithm-specific math on register data        │
│  8. Reduce: row_reduce for global statistics (optional)      │
│  9. Store: R → G (ctx.store_from_regs)                       │
└──────────────────────────────────────────────────────────────┘
```

### Example: Softmax Kernel

```python
class Softmax:
    def __init__(self, dtype, N, cluster_n, online_softmax=False):
        self.config = ElementwiseTileConfig(dtype=dtype, N=N, cluster_n=cluster_n, stage=2)

    @cute.kernel
    def kernel(self, mX, mO, tiler_mn, tiled_copy, threads_per_row):
        # ── Setup (common infrastructure) ──
        ctx = ElementwiseKernelContext(self.config)
        ctx.setup(mX, tiler_mn, tiled_copy)
        gO = ctx.partition_global(mO, tiler_mn)
        tXgO = ctx.partition_dest(gO)
        tXrO = ctx.make_fragment(tXgO)

        # ── Load ──
        ctx.load_to_smem_and_regs()

        # ── Compute (algorithm-specific) ──
        x = ctx.tXrX.load().to(cute.Float32)
        max_x = row_reduce(x, cute.ReductionOp.MAX, ...)
        exp_x = cute.math.exp2(x * log2_e - max_x * log2_e, fastmath=True)
        denom = row_reduce(exp_x, cute.ReductionOp.ADD, ...)
        y = exp_x * cute.arch.rcp_approx(denom)

        # ── Store ──
        tXrO.store(y.to(tXrO.element_type))
        ctx.store_from_regs(tXrO, tXgO)
```

### Example: RMSNorm Kernel

```python
class RMSNorm:
    def __init__(self, dtype, N, is_layernorm):
        self.config = ElementwiseTileConfig(dtype=dtype, N=N, cluster_n=None, stage=1)

    @cute.kernel
    def kernel(self, mX, mW, mB, mRes, mO, mResO, mRstd, mMean, eps, ...):
        # ── Setup (common infrastructure) ──
        ctx = ElementwiseKernelContext(self.config)
        ctx.setup(mX, tiler_mn, tiled_copy, extra_smem_tensors=1 if has_res else 0)

        # ── Load x (+ residual) ──
        vector_copy(ctx.tXgX, ctx.tXsX, is_async=True)
        # ... load residual, weight, bias ...

        # ── Reduce ──
        sum_sq = row_reduce(x * x, cute.ReductionOp.ADD, ...)
        rstd = cute.math.rsqrt(sum_sq / N + eps, fastmath=True)

        # ── Normalize ──
        y = x * rstd * weight + bias

        # ── Store ──
        tXrO.store(y.to(tXrO.element_type))
        vector_copy(tXrO, tXgO)
```

---

## Design Decisions

### Why not fully abstract the load/store?

Some kernels (like RMSNorm) need fine-grained control over the load pipeline:

- **Residual loading:** RMSNorm loads both `x` and `residual` into separate smem buffers simultaneously.
- **Delayed weight loading:** Weights can be loaded while waiting for the async copy to complete, overlapping memory latency.
- **Reload from smem:** For large \( N \), data is reloaded from smem after the reduction phase to avoid register pressure.
- **Fill OOB:** Softmax needs to fill out-of-bounds smem values with `-inf` between the G→S and S→R stages.

For these reasons, `ElementwiseKernelContext` provides both high-level helpers (`load_to_smem_and_regs`, `store_from_regs`) and low-level building blocks (`vector_copy_fn`, `tXgX`, `tXsX`, `tXrX`) so kernels can mix and match as needed.

### Why compute `tiler_mn` in `__call__` instead of `__init__`?

The `tiler_mn` and `tiled_copy` depend on `vec_size`, which in turn depends on the `largest_dtype_width` across all input/output tensors. This is only known at `@cute.jit` compile time (when the actual tensor types are resolved), not at Python construction time. Therefore, these are computed in `__call__` and passed to the kernel.

---

## Directory Structure

```
htile/
├── __init__.py          # Public API exports
├── config.py            # ElementwiseTileConfig
├── dataflow.py          # ElementwiseLoader, ElementwiseStorer, ElementwiseKernelContext
├── copy/
│   ├── __init__.py
│   └── vector.py        # VectorCopy, vector_copy, fill_oob, predicate_k
├── compute/
│   ├── __init__.py
│   └── reduce.py        # ReduceLayout, row_reduce, block_reduce, cluster_reduce
├── types/
│   ├── __init__.py
│   └── register.py      # RegTile, VectorRegTile
└── utils.py             # expand, make_fake_tensor, store_shared_remote, elem_pointer
```
