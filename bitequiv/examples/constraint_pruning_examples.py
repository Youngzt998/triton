"""Examples for constraint-aware autotuning (Starter Task T3 + T4).

These demonstrate the two new ``triton.autotune`` capabilities added in this
branch:

* **T3 — correctness gating** via ``correctness_fn`` / ``correctness_prune``:
  the autotuner runs each config once, compares its output against a
  user-supplied reference, records a success rate, and (by default) drops
  configs that produce the wrong answer before picking the fastest.

* **T4 — artifact (IR/PTX) pruning** via
  ``prune_configs_by={"artifact_config_prune": ...}``: each config is compiled
  (``run(warmup=True)`` — no launch) and its TTGIR/PTX is inspected, so configs
  can be kept/dropped by a feature present in the generated code.

Run (needs a CUDA GPU):

    python bitequiv/examples/constraint_pruning_examples.py

Each example prints what was pruned and which config won.
"""
import torch

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# T3: correctness gating on a reduction that is only correct for some configs
# ---------------------------------------------------------------------------
def example_correctness_gating():
    """A single-block sum reduction. The kernel only sums ``BLOCK_SIZE`` elements,
    so configs with ``BLOCK_SIZE < N`` silently produce a WRONG (truncated) sum.
    ``correctness_fn`` catches them; the winner is the fastest *correct* config.
    """
    N = 4096
    src = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty(1, device="cuda", dtype=torch.float32)
    ref = src.sum()

    def correctness_fn(named):
        return torch.allclose(named["dst"], ref, atol=1e-3, rtol=1e-3)

    configs = [triton.Config({"BLOCK_SIZE": bs}) for bs in (512, 1024, 2048, 4096, 8192)]

    @triton.autotune(configs=configs, key=["N"], restore_value=["dst"], correctness_fn=correctness_fn,
                     correctness_prune=True)
    @triton.jit
    def sum_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs, mask=offs < N, other=0.0)
        tl.store(dst, tl.sum(x, axis=0))

    sum_kernel[(1, )](src, out, N)
    print("[T3] correctness gating")
    for c, ok in sum_kernel.correctness_results.items():
        print(f"     BLOCK_SIZE={c.kwargs['BLOCK_SIZE']:>5}  correct={ok}")
    print(f"     winner BLOCK_SIZE={sum_kernel.best_config.kwargs['BLOCK_SIZE']} "
          f"(must be >= N={N})\n")


# ---------------------------------------------------------------------------
# T4 filter A: vectorization (PTX) — keep only configs that emit wide vector
#              global memory ops (feature selection / specialization).
# ---------------------------------------------------------------------------
def example_vectorization_filter():
    N = 1 << 20
    src = torch.randn(N, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    def keep_vectorized(config, asm, metadata):
        ptx = asm.get("ptx", "")
        # Wide vectorized loads/stores show up as ld/st.global.v4 / v2 in PTX.
        return ("ld.global.v4" in ptx) or ("st.global.v4" in ptx) or ("ld.global.v2" in ptx)

    configs = [triton.Config({"BLOCK_SIZE": bs}) for bs in (64, 128, 256, 1024, 4096)]

    @triton.autotune(configs=configs, key=["N"], prune_configs_by={"artifact_config_prune": keep_vectorized})
    @triton.jit
    def copy_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m = offs < N
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]), )
    copy_kernel[grid](src, dst, N)
    print("[T4-A] vectorization (PTX ld/st.global.v*) feature selection")
    print(f"     dropped (non-vectorized): "
          f"{sorted(c.kwargs['BLOCK_SIZE'] for c in copy_kernel.pruned_by_artifact)}")
    print(f"     winner BLOCK_SIZE={copy_kernel.best_config.kwargs['BLOCK_SIZE']}\n")


# ---------------------------------------------------------------------------
# T4 filter B: AutoWS (TTGIR) — keep only warp-specialized configs.
#              Illustrative: matches the warp-specialization marker in TTGIR.
# ---------------------------------------------------------------------------
def example_autows_filter():
    """Demonstrates filtering by a TTGIR feature. ``async_task_id`` (or a
    ``warp_specialize`` region) marks warp-specialized code. This predicate keeps
    only configs whose TTGIR shows that marker. On kernels/targets that do not
    produce warp specialization the kept set may be empty — that's expected; the
    point is the *mechanism* of TTGIR-based selection.
    """
    M, K, Nn = 512, 512, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, Nn, device="cuda", dtype=torch.float16)
    c = torch.empty(M, Nn, device="cuda", dtype=torch.float16)

    def keep_warp_specialized(config, asm, metadata):
        ttgir = asm.get("ttgir", "")
        return ("async_task_id" in ttgir) or ("warp_specialize" in ttgir) or ("ttng.warp_group" in ttgir)

    configs = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=nw, num_stages=ns)
        for nw in (4, 8)
        for ns in (2, 3)
    ]

    @triton.autotune(configs=configs, key=["M", "N", "K"],
                     prune_configs_by={"artifact_config_prune": keep_warp_specialized})
    @triton.jit
    def matmul(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + offs_m[:, None] * K + (k + offs_k)[None, :])
            b = tl.load(b_ptr + (k + offs_k)[:, None] * N + offs_n[None, :])
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(Nn, meta["BLOCK_N"]))
    try:
        matmul[grid](a, b, c, M, Nn, K)
        print("[T4-B] AutoWS (TTGIR marker) feature selection")
        print(f"     dropped (non-WS): {len(matmul.pruned_by_artifact)} configs")
        print(f"     winner: {matmul.best_config}\n")
    except Exception as e:  # noqa: BLE001 - kept set may be empty on some targets
        print(f"[T4-B] AutoWS example: no warp-specialized config kept on this target ({e})\n")


# ---------------------------------------------------------------------------
# T4 filter C: TMEM_LOAD (TTGIR) — drop configs that lower to a tmem_load op.
#              ILLUSTRATIVE: the real FA TMEM_LOAD accuracy bug repro is not in
#              OSS source (see report / open questions). This shows the *shape*
#              of a correctness-prune-by-IR filter; swap in the real repro when
#              available.
# ---------------------------------------------------------------------------
def tmem_load_correctness_filter(config, asm, metadata):
    """Return False (drop) for any config whose TTGIR contains a TMEM load op."""
    ttgir = asm.get("ttgir", "")
    return "tmem_load" not in ttgir  # keep configs WITHOUT the (hypothetically buggy) op


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("These examples require a CUDA GPU.")
    example_correctness_gating()
    example_vectorization_filter()
    example_autows_filter()
    print("[T4-C] TMEM_LOAD filter is provided as `tmem_load_correctness_filter` "
          "(illustrative; see report).")
