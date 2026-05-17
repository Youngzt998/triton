# TTGIR Ops Reference for SMT Formalization

All ops defined in `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`.

TTGIR (Triton GPU IR) is the GPU-specific lowering of TTIR. All TTIR ops are retained (with layout encoding attributes added to tensor types), and new GPU-specific ops are inserted. The validator must handle both layers.

---

## Key Difference from TTIR: Layout Encodings

Every tensor in TTGIR carries a layout encoding attribute that describes how its elements are physically distributed across threads and warps on the GPU. Two common families:

- **Distributed encodings** (e.g. `#ttg.blocked`, `#ttg.mma`) — each thread holds a subset of the tensor elements. Used for register-resident tensors.
- **Shared encodings** (e.g. `#ttg.swizzled_shared`) — elements live in shared memory. Used with `ttg.memdesc` values.

For SMT-based equivalence checking, layout encodings are **semantically transparent**: two programs that compute the same mathematical values but distribute them differently across threads are equivalent. The validator treats layout as metadata and focuses on the mathematical values.

---

## New Types in TTGIR

| Type | Mnemonic | Description |
|---|---|---|
| `MemDescType` | `ttg.memdesc<NxT, #enc, #space>` | Handle to a shared memory buffer; carries shape, element type, encoding, and memory space |
| `AsyncTokenType` | `ttg.async.token` | SSA token returned by async ops; consumed by `ttg.async_wait` |

---

## Simple Ops

### Layout Conversion

| Op | Mnemonic | SMT encoding |
|---|---|---|
| `TTG_ConvertLayoutOp` | `ttg.convert_layout` | **Identity** — same elements, different thread assignment. The mathematical values are unchanged; only the physical distribution changes. Encode as no-op: output = input. |

This is the most common op inserted by TTGIR passes. From the validator's perspective it is a no-op on values.

### Shared Memory View Ops (Pure, no memory effects)

These ops produce a new `MemDesc` pointing to a sub-region of an existing buffer. They are analogous to pointer arithmetic — they change the descriptor but do not read or write memory.

| Op | Mnemonic | SMT encoding |
|---|---|---|
| `TTG_MemDescIndexOp` | `ttg.memdesc_index` | Pointer arithmetic: advance base by `index * slice_size` |
| `TTG_MemDescSubsliceOp` | `ttg.memdesc_subslice` | Pointer arithmetic: advance base by `offsets` |
| `TTG_MemDescTransOp` | `ttg.memdesc_trans` | Permute the logical index mapping; no memory change |
| `TTG_MemDescReshapeOp` | `ttg.memdesc_reshape` | Reinterpret shape; no memory change |
| `TTG_MemDescReinterpretOp` | `ttg.memdesc_reinterpret` | Reinterpret element type and shape; analogous to `tt.bitcast` |

### Type Conversion

| Op | Mnemonic | SMT encoding |
|---|---|---|
| `TTG_Fp4ToFpOp` | `ttg.fp4_to_fp` | Unpack i8-packed fp4 pairs into fp values; encode as a bitvector extraction + conversion function under FPA mode |

### Synchronization / Side-Effect Only

For a single-threaded SMT model of one kernel instance, thread synchronization has no effect on values. These ops can be dropped.

| Op | Mnemonic | SMT encoding |
|---|---|---|
| `TTG_BarrierOp` | `ttg.barrier` | **No-op** — synchronization has no effect on values in a single-threaded model |
| `TTG_AsyncWaitOp` | `ttg.async_wait` | **No-op** — wait ensures prior async copies are complete; semantically equivalent to "the copies have happened" once wait is issued |
| `TTG_AsyncCommitGroupOp` | `ttg.async_commit_group` | **No-op** — bookkeeping for async group boundaries |

### Hardware / Program State

| Op | Mnemonic | SMT encoding |
|---|---|---|
| `TTG_WarpIdOp` | `ttg.warp_id` | Free `BitVec(32)` variable (like `tt.get_program_id`); universally quantified implicitly |
| `TTG_PredicateStageOp` | `ttg.predicate_stage` | Arithmetic on loop IV / bounds; encode as bitvector comparison |

### Structural / Pipeliner

| Op | Mnemonic | SMT encoding |
|---|---|---|
| `TTG_MaskOp` | `ttg.mask` | Conditional region; encode as `z3::ite` on predicate |
| `TTG_MaskReturnOp` | `ttg.mask.return` | Region terminator; yield values from `z3::ite` |

---

## Non-Trivial Ops

### Shared Memory Allocation / Lifetime

Shared memory in TTGIR is intermediate state — not part of the kernel's observable input/output. Its contents do not need to match between two equivalent programs as long as the final global memory writes agree.

| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TTG_LocalAllocOp` | `ttg.local_alloc` | Allocate a fresh symbolic shared memory array `Array(BV index, elem_sort)`. If an initializer (`src`) is provided, assert it equals the initial array contents. |
| `TTG_LocalDeallocOp` | `ttg.local_dealloc` | No value effect; marks the descriptor as dead. Drop in SMT. |
| `TTG_GlobalScratchAllocOp` | `ttg.global_scratch_alloc` | Allocate a fresh symbolic region in global memory (private to one program instance); treat as a fresh pointer constant. |

### Shared Memory Load / Store

These are the primary data movement ops between register tensors and shared memory. They are semantically similar to `tt.load`/`tt.store` but operate on `MemDesc` handles instead of pointer tiles.

| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TTG_LocalStoreOp` | `ttg.local_store` | Write a distributed tensor into the symbolic shared memory array. No mask — writes all elements. Analogous to an unmasked `tt.store`. |
| `TTG_LocalLoadOp` | `ttg.local_load` | Read all elements from symbolic shared memory into a distributed tensor. Analogous to an unmasked `tt.load`. |
| `TTG_LocalGatherOp` | `ttg.local_gather` | Read `result[I] = src[I[0], ..., indices[I], ..., I[n]]` — indirect indexing into shared memory. Encodes as `select` on the shared memory array at a symbolic index. |
| `TTG_LocalScatterOp` | `ttg.local_scatter` | Write `dst[..., indices[I], ...] = values[I]` — inverse of gather. Encodes as a conditional update of the shared memory array at symbolic indices. |

### Async Global-to-Local Copy

Semantically equivalent to a masked `tt.load` followed by `ttg.local_store`, with the execution deferred until `ttg.async_wait`.

| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TTG_AsyncCopyGlobalToLocalOp` | `ttg.async_copy_global_to_local` | Encode as: read bytes from global memory at `src` pointer tile (masked), write to shared memory `MemDesc`. The async token carries no value — it just establishes the happens-before edge that `ttg.async_wait` enforces. In SMT, model the copy as if it executes immediately (the wait is a no-op). |

### Warp Specialization

Warp specialization partitions the kernel's warp group into sub-groups that execute different code concurrently. Each partition region is isolated from above (no implicit captures in partition regions).

| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TTG_WarpSpecializeOp` | `ttg.warp_specialize` | Concurrently execute the default region and N partition regions on separate warp groups. All regions share global memory. For the validator, symbolically execute all regions and assert that their combined global memory effects match between the two programs. The "default" region uses the SSA results of the op (`warp_yield`). |
| `TTG_WarpSpecializePartitionsOp` | `ttg.warp_specialize.partitions` | Container for the isolated partition regions. Explicit captures replace the implicit captures of the default region. |
| `TTG_WarpYieldOp` | `ttg.warp_yield` | Terminator of the default region; passes values out as op results. Encode as `return`. |
| `TTG_WarpReturnOp` | `ttg.warp_return` | Implicit terminator of partition regions; no result values. |

Warp specialization is a major SMT challenge: it introduces true concurrency at the sub-kernel level. For the first validator milestone, the safe approach is to treat each region as a separate sequential program and assert that all combined writes to global memory agree.

---

## Summary

| Category | Ops | SMT difficulty |
|---|---|---|
| Layout conversion | `convert_layout` | Trivial (identity) |
| MemDesc views | `memdesc_index`, `memdesc_subslice`, `memdesc_trans`, `memdesc_reshape`, `memdesc_reinterpret` | Low (pointer arithmetic) |
| Sync / tokens | `barrier`, `async_wait`, `async_commit_group` | Trivial (drop) |
| Type conversion | `fp4_to_fp` | Low (bitvector op) |
| Hardware state | `warp_id`, `predicate_stage` | Low (free variable / arithmetic) |
| Shared memory alloc | `local_alloc`, `local_dealloc`, `global_scratch_alloc` | Medium (fresh symbolic array) |
| Shared memory I/O | `local_store`, `local_load`, `local_gather`, `local_scatter` | Medium (array read/write) |
| Async copy | `async_copy_global_to_local` | Medium (model as synchronous) |
| Warp specialization | `warp_specialize`, `warp_yield`, `warp_return` | High (concurrent regions) |

---

## Minimal Subset for `add_kernel` at TTGIR

A simple elementwise kernel like `add_kernel` uses no shared memory and no warp specialization. The TTGIR version of the kernel will primarily contain the original TTIR ops (with layout annotations added) and possibly some `ttg.convert_layout` insertions:

- All ops from the TTIR minimal subset (`tt.get_program_id`, `tt.make_range`, `tt.splat`, `tt.addptr`, `tt.load`, `tt.store`, `arith.*`)
- `ttg.convert_layout` — treat as identity
- `ttg.barrier` — drop

No shared memory ops appear in elementwise kernels.

---

## Relationship to TTIR Validator

The TTGIR validator builds directly on the TTIR validator:

1. **TTIR ops**: handled identically — the same `Env` and `Memory` model applies.
2. **New TTG ops**: handled by an additional translation layer in `semantics/mlir/`.
3. **Shared memory**: introduce a second `Memory` object per function (one for global, one for shared) with the same byte-addressable `Array(BV64, BV8)` model.
4. **Layout encodings**: ignored for value semantics; the validator asserts value equivalence independent of layout.
