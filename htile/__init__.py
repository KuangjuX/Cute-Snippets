"""
HTile: Hardware-Aware Tile Primitives for CuTe DSL

HTile is a collection of hardware-friendly tile primitives built on top of CuTe, designed to facilitate efficient and user-friendly development of CUDA tensor operations. The key features include:

- High-performance vector/matrix tile copy primitives and utilities, such as `VectorCopy` and `vector_copy`, enabling fast data movement, layout transformation, async/masked copy, and efficient exploitation of Hopper architecture features (like STF/CPT/cluster instructions).
- Utilities for working with cutlass.cute abstractions, including thread groups, fragments, and shared memory, to support operations such as row reduction (`row_reduce`), blockwise reduction, and flexible thread/block layouts (`ReduceLayout`) for organizing efficient reductions and tensor computations.
- Tools for generating "fake tensors" (`make_fake_tensor`), which facilitate rapid prototyping, correctness, and performance testing without requiring real data allocation.
- Tight integration with cutlass.cute and PyTorch tensor ecosystems, making it easy to plug into engineering workflows for CUDA kernel or custom operator development.
- Highly composable and extensible design, supporting advanced tile-based operator construction, such as softmax, swizzle GEMM, and various optimized kernels commonly used in Transformer workloads.
- TileFusion-inspired separation of tile configuration (`ElementwiseTileConfig`) from dataflow (`ElementwiseLoader`, `ElementwiseStorer`, `ElementwiseKernelContext`), enabling kernel developers to focus on algorithm-specific computation while reusing common infrastructure.

HTile is optimized for Hopper (SM90+) GPU architecture and is intended to be used in conjunction with CUTLASS 3.x and its cute module. For further documentation and examples, please refer to the example softmax kernel in the `kernels/` directory or explore the provided notebooks.


"""

from . import copy
from .copy import VectorCopy, vector_copy

from . import compute
from .compute import row_reduce, ReduceLayout

from . import types
from .types import RegTile, VectorRegTile

from . import utils
from .utils import (
    expand,
    make_fake_tensor,
    store_shared_remote,
    elem_pointer,
    torch2cute_dtype_map,
)

from .config import ElementwiseTileConfig
from .dataflow import (
    ElementwiseLoader,
    ElementwiseStorer,
    ElementwiseKernelContext,
)

__all__ = [
    "copy",
    "compute",
    "types",
    "utils",
    "expand",
    "VectorCopy",
    "vector_copy",
    "row_reduce",
    "ReduceLayout",
    "RegTile",
    "VectorRegTile",
    "make_fake_tensor",
    "store_shared_remote",
    "elem_pointer",
    "torch2cute_dtype_map",
    "ElementwiseTileConfig",
    "ElementwiseLoader",
    "ElementwiseStorer",
    "ElementwiseKernelContext",
]
