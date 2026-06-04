# Reference: Internal Workplace Posts on Bitwise Equivalence

Captured summaries of the four foundational Workplace posts cited in the project
plan. Content is summarized here because internal links can rot and require auth;
the originals have full images/PTX. Loaded 2026-06-03.

> **Two distinct equivalence problems run through these posts — keep them apart:**
> - **(A) Cross-config Triton equivalence** — do different autotuning configs of
>   *the same Triton kernel* agree bit-for-bit? → **this is our intern project**
>   (M1/M2; M3 extends to matching cuBLAS). Post #1 (Nick) is the direct parent.
> - **(B) Triton/compile vs PyTorch eager equivalence** — does `torch.compile`
>   output match eager? → the broader TorchInductor `strict_numerics` effort.
>   Posts #2–#4. Our project touches this only at the M5 PyTorch stretch.

---

## #1 — "Triton Bitwise Numerics: When Reductions Don't Match"
**Nick Riasanovsky · 2026-03-10 · [hhc1]** ·
groups/420659799592399/permalink/1245881820403522
*(Most directly relevant — the Triton-side framing our project extends.)*

**TL;DR:** Triton's reduction order is a **fixed algorithm whose ordering is a
byproduct of the selected layout.** Numerics change when the layout changes —
which happens via tuning/heuristics, compiler changes, or input-type changes.
Layouts aren't user-exposed. Long-term stability needs one of:
- **Fixed Reduction Ordering** — compiler guarantees order even if layout changes, or
- **Fixed Layouts** — layout won't change between runs.

**Key content:**
- Bitwise equiv definition: same input → identical bits across all candidate
  kernels (e.g. all autotuning configs). Not a promise for *arbitrary* configs.
- Tiered reduction: **thread-level** (series of `add.f32`) → **warp-level**
  (5-step tree via `shfl.sync.bfly`, offsets 16→8→4→2→1) → **cross-warp** (shared
  memory store/load + more warp reductions).
- Layout fields (`#blocked`): `sizePerThread` (vectorization / consecutive
  elems per thread), `threadsPerWarp`, `warpsPerCTA`.
- **Layout controls data placement; the reduction algorithm is fixed on the data
  held → layout changes indirectly change numerics.**
- BUT two layouts *can* agree: `num_warps=4` (`warpsPerCTA=[1,4]`) vs
  `num_warps=8` — if 8 warps give `warpsPerCTA=[2,4]` (per-row reduction
  unchanged) numerics match; if `[1,8]` (offset changes) they don't. This is why
  top-level params *inconsistently* affect numerics.
- **Experiment:** 2D sum over N. Configs {XBLOCK∈1,4}×{num_warps∈4,8}, ×
  row/col-major = 8 kernels, 28 comparisons → **only 4 match** (XBLOCK 1 vs 4,
  everything else fixed). Findings:
  - Varying XBLOCK matched *only* because layouts were identical; removing stride
    specialization made XBLOCK=4 flip to `sizePerThread=[1,1], threadsPerWarp=[4,8]`
    (parallelize loads when coalescing unavailable) → breaks.
  - Row vs col-major differ due to vectorization (`sizePerThread=[1,4]` vs `[4,1]`).
  - num_warps 4→8 → `warpsPerCTA` `[1,4]`→`[1,8]` (all warps per row) → differ.
- **Next steps:** expose explicit reduction-ordering control to users
  (benchmarking perf impact).

---

## #2 — "Chasing Bitwise Equivalent PyTorch Eager and Compile: Division, Gelu, and Reduction Ordering"
**Paul Zhang (w/ Markus Hoehnerbach) · 2025-10-07 · [LLIe]** ·
groups/257735836456307/permalink/1008914801338403
*(The deep eager-vs-Triton reduction dive — most technically detailed.)*

**Three fixes for eager↔compile parity:**
- **Division:** Triton `/` doesn't round-to-nearest by default → use
  `triton.language.div_rn`; Inductor PR to emit `div_rn` (pytorch #164144).
- **Gelu/erf:** mismatch came from **different libdevice** between Triton and
  eager. Fix: `TRITON_LIBDEVICE_PATH=$CUDA_ROOT/nvvm/libdevice/libdevice.10.bc`
  → bitwise-equal `torch.erf`/`gelu`. (Not a denormal/FTZ issue.)
- **Reductions** (the hard part):

**Reduction findings (concrete, high-value):**
- `torch.mean` on (1024, R): FP32 is **never** bitwise-equal by default; FP16/BF16
  often *are* (error masked by rounding to fewer bits). Failure rate rises with R.
- Eager reduction stages: thread-level → block-level (shared mem) → global
  (semaphore). For R ≤ 8192 a single warp suffices (max 256 elems/thread × 32).
- **Critical: shuffle direction is opposite.** Triton warp-shuffles offset
  **16→down→1**; eager (aten `Reduce.cuh`) goes offset **1→up→dim** — different
  tree, different bits. Matching eager's order in Triton → equivalence (pytorch
  #164790, WIP).
- Inductor unrolls reductions with dim < 8 (`unroll_reductions_threshold`),
  summing left-to-right; eager uses **multiple thread accumulators** for ILP
  (not left-to-right). pytorch #164755 matches eager order for R<8.
- **Vectorization:** R≥256 → 128-bit loads = 4×FP32. Triton accumulates
  vectorized loads left-to-right; eager uses 4 accumulators (`r1+r9, r2+r10,…`).
  Bitwise equiv only at **multiples of 4** (R=257..259 fail, 260/264/268 pass) —
  Triton can't vectorize non-multiples-of-4 (OOB last load); fix = pad in Inductor.
- **Looped reductions:** Inductor switches persistent→looped at R≥2048
  (interleaved accumulation) — left as future work.
- Comment thread: desire for a **Triton switch for determinism across num_warps**;
  custom pass manager (Corbin Robeck) to control layouts → more configs while
  keeping equivalence; for some shapes force 1 warp to do an independent reduction.

---

## #3 — "A path to bitwise-equivalent TorchInductor"
**Jason Ansel · 2026-02-12 · [Sk5W]** ·
groups/257735836456307/permalink/1101398988756650
*(The strategic vision: `strict_numerics` mode for TorchInductor.)*

**Proposal:** a **`strict_numerics`** TorchInductor mode producing a
**bitwise-identical result to PyTorch eager.** Driven by frontier-training needs
where validating bitwise-equal changes is far cheaper than numerical diffs.

**Catalogued sources of divergence:**
- **Reduction ordering** (biggest): RBLOCK size (share eager's block-size
  heuristic), shared-memory tree/butterfly order (may need Triton changes), split
  reductions (align algorithms).
- **Decompositions:** FMA↔unfused mul+add, division↔reciprocal-mul, intermediate
  dtypes. Verify each via OpInfo; match eager by default where possible, else
  disable under strict + fall back.
- **Precision casting:** `emulate_precision_casts` (strict implies it).
- **Compiler passes:** disable numerics-changing ones (e.g. SDPA pattern match).
- **Matmul/Attention/Convolution:** bitwise equiv across impls likely impossible
  short-term → **fall back to eager kernels** under strict.
- **Other:** functionalization; `libdevice.atan` ≠ eager `atan` (pytorch #173478);
  capturable optimizers should match.
- **Testing:** OpInfo per-op, reduction microbenchmarks (sweep shapes),
  `atol=rtol=0` patched into existing tests, full-model + accuracy minifier +
  agentic skills.
- **Scope:** target **NVIDIA B200 only** (largest training runs); other HW secondary.
- **FAQ — fusions do NOT affect numerics:** fusion just passes a value through
  registers instead of store/load; bitwise-equal as long as math ops are unchanged.

---

## #4 — "Towards Bitwise Numerical Parity Between Eager and Inductor-Compiled Optimizer Code"
**Michael Lazos · 2026-02-07 · [u2b6]** ·
groups/257735836456307/permalink/1097171219179427
*(Concrete optimizer bugfixes; Claude-assisted repros.)*

Three sources of eager↔compiled divergence found debugging the **LaProp/Adam**
optimizer (EMA accumulation amplifies tiny diffs over iterations):
1. **Implicit float64 upcasting of subnormal float32 constants** — Triton
   returned bare `tl.float32` constants subject to promotion; subnormals
   (<~1.175e-38) became float64. Fix: `tl.full({shape}, {val}, {type})`.
2. **Missing FMA lowerings** — eager `addcmul`/`addcdiv` use FMA (one rounding);
   compile used separate mul+add. Fix: register lowerings using `ops.mul_rn`
   (force-round the product, prevent auto-fusion) + `ops.fma`.
3. **Subnormal handling in FMA** — `libdevice.fma` does Flush-To-Zero; CUDA
   native FMA preserves subnormals. Fix: switch to `tl.fma`. (Later: an inductor
   config disables libdevice FTZ to match eager.)
- Verified with `atol=0, rtol=0` over edge cases (subnormals, near-boundary,
  non-representable decimals). **Claude accelerated the repros.**
- Future: make all compiled optimizers bitwise-equal to eager this half.

---

## Cross-cutting takeaways for our project

- **Layout ⇒ order ⇒ bits.** Confirms the core model: order is a byproduct of
  layout; changing layout (via config) changes numerics. (Post #1, #2.)
- **Concrete order facts to encode in tooling:**
  - Triton warp shuffle = `bfly`, offset **16→1** (decreasing). Eager = offset
    **1→up** (increasing). Opposite trees. (#1, #2)
  - Vectorization equivalence requires **multiples of 4** for FP32 (128-bit loads). (#2)
  - Persistent vs looped reduction boundary ≈ R=2048; split reductions at small R. (#2, #3)
- **Non-reduction sources we must also watch:** FMA vs mul+add (`mul_rn`/`fma`/
  `tl.fma` vs `libdevice.fma`), `div_rn`, libdevice version, FTZ/subnormals,
  dtype/precision casts, decompositions. (#2, #3, #4)
- **"Two solutions" framing (Post #1) = our project's levers:** fixed reduction
  ordering (enforce, `inner_tree`) vs fixed layouts (freeze). We pursue *enforce +
  optimize* to beat *freeze*.
- **Half-precision can mask differences** — test FP32 to actually surface order
  bugs; FP16/BF16 false-passes. (#2)
- **Hard cases fall back / can't be made equivalent:** matmul/attn/conv across
  impls (#3) — mirrors our M3 caveat that some configs can't honor the order.
- **Testing discipline:** `atol=rtol=0`, OpInfo, microbenchmark shape sweeps,
  accuracy minifier — directly applicable to our experiment framework (#3, #4).
- **People:** Nick (Triton reductions/layout), Paul Zhang + Markus (eager↔Triton
  reduction deep-dive), Jason Ansel (strict_numerics strategy), Michael Lazos
  (optimizer fixes), Corbin Robeck (custom pass manager for layout control).
