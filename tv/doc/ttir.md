# TTIR Ops Reference for SMT Formalization

All ops defined in `include/triton/Dialect/Triton/IR/TritonOps.td`.

---

## Simple Ops

Direct SMT/bitvector encoding, no side effects or complex control flow.

### Cast / Type Conversion
| Op | Mnemonic | Description |
|---|---|---|
| `TT_IntToPtrOp` | `tt.int_to_ptr` | i64 → pointer (identity on bitvector) |
| `TT_PtrToIntOp` | `tt.ptr_to_int` | pointer → i64 |
| `TT_BitcastOp` | `tt.bitcast` | Reinterpret bits, no value conversion |
| `TT_FpToFpOp` | `tt.fp_to_fp` | Float precision change (needs FP encoding) |

### Pointer Arithmetic
| Op | Mnemonic | Description |
|---|---|---|
| `TT_AddPtrOp` | `tt.addptr` | `ptr + offset`, elementwise on tensors |
| `TT_AdvanceOp` | `tt.advance` | Advance block pointer by strides (structured pointer) |

### Tensor Construction
| Op | Mnemonic | Description |
|---|---|---|
| `TT_SplatOp` | `tt.splat` | Scalar → tensor (broadcast scalar to all elements) |
| `TT_UnsplatOp` | `tt.unsplat` | Tensor → scalar (inverse of splat, 1-element) |
| `TT_MakeRangeOp` | `tt.make_range` | `[start, end)` index range tensor |
| `TT_ExpandDimsOp` | `tt.expand_dims` | Insert a size-1 dimension |
| `TT_BroadcastOp` | `tt.broadcast` | Broadcast along 1-dims |
| `TT_ReshapeOp` | `tt.reshape` | Reinterpret shape, same elements |
| `TT_CatOp` | `tt.cat` | Concatenate two tensors |
| `TT_JoinOp` | `tt.join` | Interleave two tensors into higher dim |
| `TT_SplitOp` | `tt.split` | Inverse of join |
| `TT_TransOp` | `tt.trans` | Transpose dimensions (permutation) |

### Arithmetic / Elementwise
| Op | Mnemonic | Description |
|---|---|---|
| `TT_ClampFOp` | `tt.clampf` | Clamp float to [min, max] |
| `TT_PreciseSqrtOp` | `tt.precise_sqrt` | IEEE sqrt |
| `TT_PreciseDivFOp` | `tt.precise_divf` | IEEE division |
| `TT_MulhiUIOp` | `tt.mulhiui` | High bits of u64 multiply |

### Program ID
| Op | Mnemonic | Description |
|---|---|---|
| `TT_GetProgramIdOp` | `tt.get_program_id` | Block index along axis (free variable in SMT) |
| `TT_GetNumProgramsOp` | `tt.get_num_programs` | Grid size (free variable in SMT) |

### Control Flow / Structural
| Op | Mnemonic | Description |
|---|---|---|
| `FuncOp` | `tt.func` | Function definition |
| `CallOp` | `tt.call` | Direct function call |
| `ReturnOp` | `tt.return` | Function return |
| `TT_ReduceReturnOp` | `tt.reduce.return` | Region terminator for reduce |
| `TT_ScanReturnOp` | `tt.scan.return` | Region terminator for scan |
| `TT_MapElementwiseReturnOp` | `tt.map_elementwise.return` | Region terminator for map_elementwise |

### Debug / Side-effect Only
| Op | Mnemonic | Description |
|---|---|---|
| `TT_PrintOp` | `tt.print` | No semantic effect on values |
| `TT_AssertOp` | `tt.assert` | Adds a path constraint (maps to SMT `assert`) |

---

## Non-Trivial Ops

Require careful modeling: memory model, quantifiers, or axiomatization.

### Memory Ops — require array theory or separation logic
| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TT_LoadOp` | `tt.load` | Masked load: `∀i. mask[i] ? mem[ptr[i]] : other[i]` |
| `TT_StoreOp` | `tt.store` | Masked store: `∀i. mask[i] → mem'[ptr[i]] = val[i]` |
| `TT_AtomicRMWOp` | `tt.atomic_rmw` | Read-modify-write with atomicity |
| `TT_AtomicCASOp` | `tt.atomic_cas` | Compare-and-swap; non-deterministic under races |

### Reduction — requires quantifier or fold encoding
| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TT_ReduceOp` | `tt.reduce` | Arbitrary combiner over a tensor dimension; combiner is a region (sub-function) |

### Scan — prefix reduction, order-sensitive
| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TT_ScanOp` | `tt.scan` | Parallel prefix with arbitrary combiner; encodes as a sequence of dependent steps |

### Gather / Histogram — indirect indexing
| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TT_GatherOp` | `tt.gather` | `out[i] = src[indices[i]]`; array reads at symbolic indices |
| `TT_HistogramOp` | `tt.histogram` | Scatter-count into bins; aliasing makes quantifier-free encoding hard |

### Matrix Multiply
| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TT_DotOp` | `tt.dot` | Tensor contraction (matmul); sum-of-products, O(n³) terms naively |
| `TT_DotScaledOp` | `tt.dot_scaled` | Dot with per-element scaling factors (MX format numerics) |

### User-defined / External Combiners — must be axiomatized
| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TT_MapElementwiseOp` | `tt.map_elementwise` | User-defined combiner region applied elementwise |
| `TT_ExternElementwiseOp` | `tt.extern_elementwise` | External C function; must be axiomatized |
| `TT_ElementwiseInlineAsmOp` | `tt.elementwise_inline_asm` | PTX inline asm; must be axiomatized |

### Tensor Descriptor / Block Pointer — complex address computation
| Op | Mnemonic | SMT challenge |
|---|---|---|
| `TT_MakeTensorPtrOp` | `tt.make_tensor_ptr` | Block pointer with shape/strides/offsets |
| `TT_MakeTensorDescOp` | `tt.make_tensor_descriptor` | TMA descriptor (hardware-managed) |
| `TT_DescriptorLoadOp` | `tt.descriptor_load` | TMA load |
| `TT_DescriptorStoreOp` | `tt.descriptor_store` | TMA store |
| `TT_DescriptorReduceOp` | `tt.descriptor_reduce` | TMA reduce |
| `TT_DescriptorGatherOp` | `tt.descriptor_gather` | TMA gather |
| `TT_DescriptorScatterOp` | `tt.descriptor_scatter` | TMA scatter |

---

## Summary

| Category | Count |
|---|---|
| Simple | ~25 |
| Non-trivial | ~15 |
| **Total** | **~40** |

## Minimal Subset for `add_kernel`

The ops needed to verify a basic kernel like `add_kernel`:

- `tt.get_program_id`
- `tt.make_range`
- `tt.splat`
- `tt.addptr`
- `tt.load`
- `tt.store`
- `tt.broadcast`
- `tt.expand_dims`
- `arith` dialect: `muli`, `addi`, `cmpi`, `addf`, `constant`
