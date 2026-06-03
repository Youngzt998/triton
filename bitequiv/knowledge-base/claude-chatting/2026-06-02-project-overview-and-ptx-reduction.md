# 2026-06-02 — Project Overview, PTX Tree Reduction, and Codebase Exploration

## What we covered

### 1. Project Plan Review
- Read the full 19-page intern project plan PDF (Bitwise Equivalence and Triton Autotuning)
- Summarized the milestone structure: Starter Tasks → M1 (reduction detection) → M2 (layout optimization) → M3 (GEMM) → Stretch goals
- Discussed Claude's capabilities vs limitations on this project (can write code/tooling/analysis, cannot run GPU kernels or profile)

### 2. Project Plan Interpretation
- Clarified that PTX-level equivalence checking is the **verification tool**, not the main goal
- The main engineering focus is **compiler passes at the TTGIR/MLIR level** (layout optimization, constraint representation) and **autotuner integration**
- The plan addresses multiple sources of bitwise non-equivalence beyond reduction ordering: MMA/tensor core instructions, Split-K, FMA vs separate mul+add, fusion, vectorization, known hardware bugs (TMEM_LOAD)

### 3. Codebase Search — What Exists vs What's New
Searched `fbexperimental_triton` and `fbsource/third-party/triton/beta` for existing infrastructure.

**Already exists:**
- `inner_tree` / `reduction_ordering` (D100027220, Nick Riasanovsky) — full compiler implementation from Python API through MLIR to PTX. Uses count-up warp shuffles (1,2,4,8,16) and balanced binary tree within threads
- `TRITON_STRICT_REDUCTION_ORDERING` env var (D101872700)
- TritonParse — IR analysis framework with PTX parsing and multi-level diff, but text-level only (no semantic analysis)
- Autotuner hooks: `early_config_prune`, `restore_value`, `TRITON_SPILL_THRESHOLD`, `CompiledKernel.metadata`, IR/PTX dump per config
- Ad-hoc repro scripts: `triton_repro_bitwise.py` (Paul Zhang)
- Production workaround: `STABLE_REDUCTION` in layer_norm (D104785121) — manual sequential loop, motivated by real 5.24% NE gap in Threads model
- PyTorch CUDA `ReduceInnerTree.cuh` — separate but related effort

**Needs to be built first-ever:**
- PTX semantic analysis tooling (reconstruct reduction tree from PTX)
- Cross-config bitwise equivalence checker (static, without GPU execution)
- Autotuner output correctness comparison during config selection
- IR-based equivalence pruning
- Layout optimization pass for ordered reductions (M2)
- GEMM/MMA constraint representation and lowering (M3)
- Systematic experiment framework

### 4. PTX Tree Reduction Deep Dive
- Explained sequential vs tree reduction and why FP non-associativity makes the tree shape matter
- Walked through simple (8-element) and complex (128-element cross-warp) examples with full PTX code
- Explained how layouts change the reduction tree even with identical PTX instruction patterns
- Created knowledge-base doc: `tree-reduction-in-ptx-and-triton.md`

### 5. How PTX Controls Reduction Order
- PTX is per-thread (SIMT model), but `shfl.sync` is a **collective operation** — all warp threads execute it simultaneously
- The reduction tree is an emergent property of: (a) which thread holds which data (layout), and (b) the shuffle offset sequence
- Count-down offsets (16,8,4,2,1) vs count-up offsets (1,2,4,8,16) produce different trees on the same data
- `inner_tree` controls order via: count-up shuffle sequence + canonical within-thread register grouping
- Different configs → different PTX files → same mathematical tree (when inner_tree works correctly)

### 6. Autotuner and PTX Relationship
- Each autotuning config compiles to a completely different PTX file
- Without `reduction_ordering`: no bitwise equivalence expected
- With `inner_tree`: bitwise equivalence is the goal, but no tooling exists to verify it statically
- The project builds static PTX analysis to verify equivalence without GPU execution, then integrates into the autotuner to prune non-equivalent configs

## Key diffs and file paths referenced
- D100027220 — `inner_tree` implementation (Nick Riasanovsky)
- D101872700 — `TRITON_STRICT_REDUCTION_ORDERING` env var
- D104785121 — `STABLE_REDUCTION` production workaround
- D75976113 — FBGEMM split of correctness vs performance pruning
- D100024902 — `triton_repro_bitwise.py` (Paul Zhang)
- `fbsource/third-party/triton/beta/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp` — inner_tree lowering
- `fbsource/third-party/triton/beta/triton/python/triton/language/core.py` — ReductionOrdering API
- `fbcode/pytorch/tritonparse/tritonparse/` — TritonParse framework

## Documents created
- `knowledge-base/tree-reduction-in-ptx-and-triton.md` — comprehensive teaching doc on tree reduction
- `knowledge-base/claude-chatting/` — this chat summary folder (new convention)

## Open questions for follow-up
- What exactly does the TMEM_LOAD accuracy bug look like?
- What was decided at the Nvidia collaboration day meeting?
- Should the PTX analysis approach be purely static, LLM-assisted, or both?
- How does TLX's API surface differ from standard Triton at the IR level?
