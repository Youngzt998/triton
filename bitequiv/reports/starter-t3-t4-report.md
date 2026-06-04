# Starter Task T3 + T4 — Implementation Report

**Branch:** `bitequiv-starter-claude-full-auto`
**Author:** Claude (autonomous run), for Ziteng Yang
**Scope:** Starter Task T3 (autotuner correctness-check hook) and T4 (IR/PTX-based
config pruning). Non-trivial starter items only — T1 (notes) and T2 (10–15 tutorial
kernels) are out of scope for this branch.

---

## 1. TL;DR

Added two **opt-in, first-class** gates to the Triton autotuner so it can reject
*incorrect* or *unwanted* configs before choosing the fastest one — the reusable
foundation that M1 (reduction equivalence) and M3 (GEMM) will build on:

- **T3 — `correctness_fn` / `correctness_prune`:** run each config once (untimed),
  compare its output to a user reference, record a per-config success rate, and by
  default drop failing configs from selection.
- **T4 — `prune_configs_by={"artifact_config_prune": ...}`:** compile each config
  (no launch) and inspect its **TTGIR/PTX**, keeping/dropping configs by a feature
  in the generated code.

Both default to **off** → zero behavior change for existing kernels.

**Validation status:**
- ✅ **7/7 CPU logic tests pass** — full T3/T4 control flow validated
  deterministically with a mock kernel (no GPU needed).
- ⏳ **2 GPU end-to-end tests written and ready**, currently **skipped**: the
  GB200 NVLink **fabric was still initializing** (`Fabric State: In Progress`,
  CUDA Error 802) for the entire session, so no kernel could launch. Re-run the
  one command in §6 once `torch.cuda.is_available()` is True.

---

## 2. What was built

### T3 — Correctness-check hook
New `@triton.autotune` parameters:
- `correctness_fn(named_args) -> bool` — receives the full arg+meta dict **after**
  one run of the config (so `named_args["<output>"]` holds that config's result);
  returns True if acceptable. The user owns the reference and comparison (e.g.
  `torch.equal(named["dst"], ref)` for bitwise, or `torch.allclose(...)`).
- `correctness_prune: bool = True` — fail ⇒ excluded from selection; `False` ⇒
  record-only (results kept, nothing dropped).

Results are exposed on the autotuner as `self.correctness_results: {Config: bool}`
for success-rate inspection, and printed when `TRITON_PRINT_AUTOTUNING=1`.

### T4 — Artifact (IR/PTX) pruning hook
New key in `prune_configs_by`:
- `artifact_config_prune(config, asm, metadata) -> bool` — return True to **keep**.
  `asm` is the `CompiledKernel.asm` dict (`ttir`/`ttgir`/`llir`/`ptx` text),
  `metadata` is `CompiledKernel.metadata`.

Dropped configs are recorded on `self.pruned_by_artifact: {Config: reason}`.

---

## 3. Design decisions and why

1. **First-class core hooks (not external helpers).** Both gates live in
   `Autotuner` so they are reusable by M1/M3 and match the project goal of tooling
   "callable directly from the autotuner." Confirmed with the user.

2. **T4 needs a *post-compile* hook — the key structural insight.**
   `early_config_prune` runs **before any compilation** (it only sees configs +
   args), so it fundamentally cannot inspect IR/PTX. T4 therefore compiles each
   config first. It uses `self.fn.run(..., warmup=True)` with the **real call
   args**, which:
   - compiles with **accurate specialization** (so the inspected PTX — including
     vectorization width, which depends on alignment/divisibility — matches what
     the launched kernel will actually use). This is strictly better than
     `JITFunction.warmup`, which wraps args in `MockTensor` and loses that
     specialization.
   - **does not launch** the kernel (`run` skips the launch when `warmup=True`).
   - **populates the JIT cache**, so the subsequent benchmark reuses the compile
     instead of recompiling — the compile cost is amortized, not doubled.

3. **Correctness check runs *outside* `do_bench`.** It is a separate single run so
   the comparison/copy cost never pollutes timing measurements.

4. **Reuse existing `restore_value`/`reset_to_zero` plumbing.** The correctness
   run mutates output buffers; the autotuner already has battle-tested
   snapshot/restore (`pre_hook`/`post_hook`). The check snapshots before the run
   and restores after, so neither benchmarking nor the final winner launch sees a
   mutated tensor. No new buffer-management code.

5. **Record-then-prune default for T3.** Always record pass/fail + success rate
   (the plan's "inspect the success rate"); by default also exclude failers so the
   winner is guaranteed correct. `correctness_prune=False` gives pure diagnostics.

6. **Fail-safe semantics.** A config that cannot compile/run is treated as failing
   (T3) or dropped-and-recorded (T4) — it could never be a valid winner anyway. If
   *every* config is pruned, a clear `AutotunerError` is raised (mirrors the
   existing `early_config_prune` empty-result behavior).

7. **String-predicate filters as the starting point.** The three T4 examples match
   substrings in TTGIR/PTX. This is backend-agnostic and simple; M1's semantic PTX
   tree analyzer can later slot behind the *same* hook unchanged.

---

## 4. Files changed

| File | Change |
|---|---|
| `python/triton/runtime/autotuner.py` | T3 + T4 implementation (see §5) |
| `python/test/unit/runtime/test_autotuner_constraint_prune.py` | **new** — 7 CPU logic tests + 2 GPU e2e tests |
| `bitequiv/examples/constraint_pruning_examples.py` | **new** — runnable examples: T3 gating, T4 vectorization/AutoWS/TMEM_LOAD filters |

### Code walk-through of `autotuner.py`
- **`__init__`** — new params `correctness_fn`, `correctness_prune`; parse
  `artifact_config_prune` from `prune_configs_by`; init `correctness_results` and
  `pruned_by_artifact` (placed before the deprecated-args early-returns so they are
  always set).
- **`_check_correctness(*args, config, **meta)`** — builds `full_nargs` like
  `_bench`, snapshots via `pre_hook`, runs the kernel once, calls `correctness_fn`,
  restores via `post_hook`; returns the bool (catches `OutOfResources` /
  `CompileTimeAssertionFailure` / `PTXASError` as failures).
- **`_artifact_prune_configs(configs, kwargs)`** — for each config:
  `run(..., warmup=True)` → inspect `kernel.asm` / `kernel.metadata` via the
  predicate; keep or record in `pruned_by_artifact`; compile failures are dropped.
- **`prune_configs`** — calls `_artifact_prune_configs` after `early_config_prune`,
  before `perf_model`; raises if nothing survives.
- **`run` → `benchmark()`** — before timing, runs the T3 check over `pruned_configs`,
  records `correctness_results`, prints success rate (when enabled), and (if
  pruning) benches only the valid set.
- **`autotune` decorator** — forwards the two new params.

---

## 5. Test results

```
$ python -m pytest python/test/unit/runtime/test_autotuner_constraint_prune.py -q
7 passed, 2 skipped   # 2 skipped = GPU e2e (no CUDA this session)
```

CPU logic tests (deterministic, mock kernel):
- `test_correctness_prune_excludes_wrong_configs` — wrong configs (BLOCK_SIZE<N)
  recorded False and excluded; fastest *correct* config wins.
- `test_correctness_record_only_does_not_prune` — results recorded, nothing pruned.
- `test_correctness_prune_all_fail_raises` — clean `AutotunerError`.
- `test_artifact_prune_filters_on_ttgir` — keeps only the config whose TTGIR
  matches; others recorded in `pruned_by_artifact`.
- `test_artifact_prune_on_metadata` — filter on `metadata.num_warps`.
- `test_artifact_prune_all_pruned_raises` — clean `AutotunerError`.
- `test_no_hooks_is_unchanged_behavior` — baseline unchanged; result dicts empty.

GPU e2e tests (ready, pending fabric):
- `test_gpu_correctness_prune_partial_sum` — real single-block sum; only
  BLOCK_SIZE≥N is correct; winner must be correct.
- `test_gpu_artifact_prune_on_real_ir` — asserts real TTGIR/PTX are present and
  inspectable; prunes by `metadata.num_warps`.

Lint/format: `ruff` (v0.9.1) clean; `yapf` (v0.43.0) applied — both pinned to the
repo's `.pre-commit-config.yaml` revisions.

---

## 6. How to use / re-validate

Run the GPU tests once CUDA is up:
```bash
cd /home/youngzt/bitwise-equiv/triton
python -c "import torch; assert torch.cuda.is_available()"   # must be True
python -m pytest python/test/unit/runtime/test_autotuner_constraint_prune.py -q
python bitequiv/examples/constraint_pruning_examples.py
```

Usage sketch (T3):
```python
@triton.autotune(configs=cfgs, key=["N"], restore_value=["out"],
                 correctness_fn=lambda a: torch.equal(a["out"], ref))
@triton.jit
def kernel(...): ...
# after run: kernel.correctness_results -> {Config: bool}
```

Usage sketch (T4):
```python
@triton.autotune(configs=cfgs, key=["N"],
                 prune_configs_by={"artifact_config_prune":
                                   lambda c, asm, md: "ld.global.v4" in asm["ptx"]})
@triton.jit
def kernel(...): ...
# after run: kernel.pruned_by_artifact -> {Config: reason}
```

---

## 7. Limitations & open items

- **GPU end-to-end validation pending.** The GB200 NVLink fabric never finished
  initializing this session (`nvidia-imex` active, `nvidia-fabricmanager` inactive,
  `Fabric State: In Progress`, CUDA Error 802). All GPU-dependent validation is
  written and ready; only the launch step is blocked by infra. **Action:** re-run
  §6 once the fabric converges (or a fabric-manager/imex restart).
- **TMEM_LOAD filter is illustrative.** The real FA TMEM_LOAD accuracy-bug repro is
  not present in OSS source (likely Meta-internal). The provided
  `tmem_load_correctness_filter` matches the `tmem_load` op and shows the correct
  *shape* of a correctness-prune-by-IR filter; swap in the real repro when
  available (open question — confirm with Nick / collab-day notes).
- **T4 compile cost.** Artifact pruning compiles every surviving config up front.
  This is amortized (cache reused at bench time) but front-loads compilation; for
  very large config sweeps consider combining with `early_config_prune` first.

---

## 8. Suggested next steps

1. Run the GPU tests + examples once the fabric is up; paste results into this
   report.
2. Wire the real TMEM_LOAD repro into the T4 filter (Starter T4 headline target).
3. M1 hook-up: implement the semantic PTX reduction-tree checker behind
   `artifact_config_prune` (and/or `correctness_fn` against a reference ordering).
