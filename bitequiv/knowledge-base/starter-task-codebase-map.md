# Starter Task — Codebase Map (files to understand)

Every file relevant to completing the Starter Task (T1–T4), tiered by how deeply
you must understand it. Paths are absolute-from-repo-root. Companion to
`starter-task-overview.md` (the goal/breakdown). Verified by codebase exploration
2026-06-03.

Legend: ★★★ understand deeply / modify · ★★ read carefully · ★ reference as needed.

---

## TIER A — Autotuner & runtime (the core of T3 + T4)

### ★★★ `python/triton/runtime/autotuner.py`  (the file you modify)
- `Autotuner.run` (~L307): cache lookup → `prune_configs(kwargs)` (L320) →
  `benchmark()` closure → winner = `min(timings)` (L347, **fastest wins**) →
  cache + run winner. Optional best-config IR dump (L369–388).
- `Autotuner._bench` (L138–178): compiles+measures one config; calls `self.fn.run`
  (L160) then `self.do_bench` (L174). **Correctness-check hook (T3) slots here**,
  after the run, before/around timing.
- `Autotuner.prune_configs` (L392–417): applies `early_config_prune` (L394–398)
  then `perf_model`/`top_k`. **Artifact-based prune (T4) plugs in via a custom
  `early_config_prune`, or a new late-prune step that compiles then inspects.**
- `Config` (L432): `kwargs`, `num_warps/num_ctas/num_stages`, `maxnreg`,
  `pre_hook`, `ir_override`; `all_kwargs()` (L522). AutoWS fields
  `minRegAutoWS/maxRegAutoWS/pingpongAutoWS` (L477–500).
- `pre_hook/post_hook` + `restore_value`/`reset_to_zero` (L42–77): existing
  reset/restore plumbing to **reuse for the correctness check** (snapshot/restore
  output buffers across configs).
- `autotune` decorator (L576) + `prune_configs_by` schema
  `{early_config_prune, perf_model, top_k}`.

### ★★ `python/triton/runtime/jit.py`
- `JITFunction.run` (L785–973) → `_do_compile` (L1079–1108) returns a
  `CompiledKernel`, cached in `device_caches[device][0]` (dict: key→CompiledKernel).
  This is **how you get the compiled artifact for a config** post-run.

### ★★ `python/triton/compiler/compiler.py`
- `CompiledKernel` (L501): `.asm` (`AsmDict`, L481) keyed `ttir|ttgir|llir|ptx|
  cubin|sass`; `.metadata` (L515; num_warps/stages/ctas/shared/tmem_size…);
  `.metadata_group` (L528: filename→on-disk path). **T4 reads `.asm['ttgir']` /
  `.asm['ptx']`; T2 uses the same to show per-stage IR.**
- `compile()` (L267–454); per-config on-disk dump dir logic (L306–327).

### ★ `python/triton/knobs.py`
- `compilation.dump_ir` = `TRITON_KERNEL_DUMP` (L358); `autotuning.dump_best_config_ir`
  = `TRITON_KERNEL_DUMP_BEST_CONFIG` (L378); `cache.dump_dir` = `TRITON_DUMP_DIR`
  (L345); `language.strict_reduction_ordering` = `TRITON_STRICT_REDUCTION_ORDERING`
  (L498). Autotuning knobs: `cache/print/warmup/rep`.

### ★ `python/triton/testing.py`
- `do_bench` (L127–190; used by `_bench`), `assert_close` (L193). Bench/report infra.

---

## TIER B — Reduction ordering end-to-end (canonical example; bridges to M1)

This is the reference "constraint" the whole project generalizes — understand the
full path even though the Starter only lightly touches it.

- ★★ `python/triton/language/core.py`: `ReductionOrderingBase` (L26),
  `ReductionOrdering` (L39), `.UNORDERED`/`.INNER_TREE` (L59–60),
  `CompositeReductionOrdering` (L63), `reduce()` (L2764; defaults to INNER_TREE
  iff `strict_reduction_ordering`, else UNORDERED).
- ★★ `python/triton/language/semantic.py`: `reduction()` (L1926) passes
  `reduction_ordering.name` (a **string**) to `create_reduce()`.
- ★ `include/triton/Dialect/Triton/IR/TritonOps.td`: `TT_ReduceOp`, attribute
  `OptionalAttr<StrAttr>:$reduction_ordering` (L765) — the attr that carries order
  on the op in TTIR/TTGIR.
- ★ `lib/Dialect/Triton/IR/Ops.cpp`: `ReduceOp::hasDefinedOrdering()` (L605) —
  present and != "unordered".
- ★★ `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`: `isInnerTree` (L95,
  matches attr value `"inner_tree"`), `reduceValueSequence` (L121: tree vs
  sequential), `warpReduce` (L247; count-up shuffle 1,2,4,…).
- ⚠️ **Naming**: Python = enum `ReductionOrdering.INNER_TREE`; on the IR op +
  C++ it's the **string** `"inner_tree"`. Don't conflate.

---

## TIER C — The three filter targets (T4)

| Target | Where it shows up | What a filter matches |
|---|---|---|
| **Reduction ordering** | `tt.reduce` attr `reduction_ordering` ("inner_tree"/"unordered") | TTGIR attr presence/value |
| **AutoWS** | op attr `async_task_id` (`DenseI32ArrayAttr`) | TTGIR: any op carrying `async_task_id` |
| **Vectorization** | `BlockedEncoding.sizePerThread[last]` → PTX `ld/st.global.v{2,4,8}` | PTX: vector load/store width |
| **TMEM_LOAD (bug)** | `ttng.tmem_load` op (+ optional `redOp`) | TTGIR op / PTX `tcgen05.ld[.red]` |

Files:
- **AutoWS**: `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/`
  (`AutomaticWarpSpecialization.cpp`, `Partition*.cpp`);
  `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/`
  (`PartitionSchedulingMeta.cpp`, `TaskIdPropagation.cpp`, `Utility.cpp` L17/47 —
  read/set `async_task_id`). TLX surface: `tlx.async_task(s)`.
  → Also load the `autows-docs` / `partition-scheduler` skills before editing.
- **Vectorization**: `lib/Analysis/AxisInfo.cpp` (alignment/contiguity → safe
  width); `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` (L37–39 width =
  min(elemsPerThread, 128/bitwidth)); `CoalesceUtils.cpp`;
  `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp` (emits vector ld/st).
- **TMEM**: `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td`
  — `TTNG_TMEMLoadOp` (L911, `redOp` L959), `TTNG_TMEMStoreOp` (L1011),
  `TTNG_TMEMAllocOp` (L1042). NOTE: literal "TMEM_LOAD" absent in OSS — the bug
  repro is likely Meta-internal; **confirm exact repro** (open question).

---

## TIER D — Examples + tests (T2 + how to validate everything)

### Where example/tutorial kernels live (T2 lands 10–15 here)
- `python/tutorials/` — numbered `NN-name.py`. Canonical structure: docstring →
  imports → `@triton.jit` kernel → Python wrapper → correctness check → optional
  `@triton.testing.perf_report` bench. Ref: `01-vector-add.py`,
  `02-fused-softmax.py`; **`06-fused-attention.py` has a real `early_config_prune`
  example** (`prune_invalid_configs`).
- `third_party/tlx/tutorials/` — TLX variants (e.g. `vector-add2.py` with
  `tlx.async_tasks()`); `@pytest.mark.skipif` for hardware.
- `python/examples/gluon/` — Gluon (do NOT modify Gluon).

### Tests
- ★★ `python/test/unit/runtime/test_autotuner.py`: `test_prune_configs`
  (L131–174) — exact `early_config_prune(configs, named_args, **kwargs)` contract
  + `do_bench`. **Model new T3/T4 tests on this.**
- ★★ `python/test/unit/language/test_core.py`: reduction-ordering bitwise tests
  — `test_reduction_ordering_sum` (L3155), `_reduce_mul` (L3195), `_argmin`
  (L3236), `_sum_multi_group` (L3276). Pattern: loop `num_warps in [1,2,4,8]`,
  `tl.ReductionOrdering.INNER_TREE`, `torch.equal(out, reference)`.
- Reference data: `python/test/unit/language/test_data/reduction_ordering_*.pt`.
- TLX correctness harness: `third_party/tlx/tutorials/testing/test_correctness.py`
  (Config classes w/ `SHAPES`/`CONFIGS`/`create_inputs`/`get_reference`;
  parametrized; `torch.testing.assert_close`).
- Bitwise-equality idioms: `torch.equal(a,b)` or `assert_close(...,atol=0,rtol=0)`;
  controlled-precision inputs via `random_bfloat16` (`test_pipeliner.py` L371).

---

## TIER E — Numerics-modifying passes (T2 doc-gap inventory)

Passes that can change bits → candidates to document:
- FMA contraction: `lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/FMA.cpp`
  (`multiplyVectors` → `LLVM::FMulAddOp` L36), `FMADotUtility.cpp`.
- `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp` (tree vs sequential — Tier B).
- `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp` (changes layout →
  changes reduction tree / vectorization).
- `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` + `lib/Analysis/AxisInfo.cpp`
  (vector width).
- `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` (MMA selection — M3).
- `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp` (TMEM vec width).
- `lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp` (libdevice / div_rn /
  exp2 / FTZ-denormal lowering).
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp`,
  `ReduceDataDuplication.cpp`, `OptimizeThreadLocality.cpp` (scheduling/locality).

---

## Quick "where do I start" per sub-task
- **T1** → `ir-debugging` skill + `knobs.py` dump vars + Tier B to read real IR.
- **T2** → Tier D (tutorials/ structure) + Tier E (passes to document) + `.asm` access.
- **T3** → `autotuner.py` `_bench`/`run` + `pre_hook`/`restore_value` + `test_autotuner.py`.
- **T4** → `autotuner.py` `prune_configs`/`early_config_prune` + `CompiledKernel.asm`
  + Tier C (the 3 targets) + `06-fused-attention.py` prune example.
