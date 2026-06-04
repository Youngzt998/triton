"""Tests for constraint-aware autotuning: correctness checking (T3) and
artifact/IR-PTX-based pruning (T4).

This file has two layers:

* **CPU logic tests** (no GPU) drive ``Autotuner.run`` end-to-end with a *mock*
  JIT function so the new control flow (correctness gating, success-rate
  recording, artifact pruning, compile-error handling) is validated
  deterministically without a GPU.
* **GPU end-to-end tests** (skipped without CUDA) exercise the same hooks through
  real Triton compilation + launch, modeled on ``test_autotuner.py``.
"""
import pytest
import torch

import triton
import triton.language as tl
from triton.runtime.autotuner import Autotuner, Config, AutotunerError

# Disk caching would need a real backend/driver; force it off for the CPU tests.
triton.knobs.autotuning.cache = False


def is_cuda():
    return torch.cuda.is_available() and triton.runtime.driver.active.get_current_target().backend == "cuda"


# ---------------------------------------------------------------------------
# CPU logic tests (mock JIT function, no GPU required)
# ---------------------------------------------------------------------------
class _FakeKernel:
    """Stand-in for triton.compiler.CompiledKernel: just carries asm + metadata."""

    class _Meta:

        def __init__(self, num_warps):
            self.num_warps = num_warps

    def __init__(self, block_size, num_warps):
        # IR text that varies by config the way real TTGIR encodes tile shapes,
        # so artifact predicates can match on it exactly like real IR.
        self.asm = {
            "ttir": f"module {{ %0 = tt.make_range tensor<{block_size}xi32> }}",
            "ttgir": f"module attributes {{\"ttg.num-warps\" = {num_warps} : i32}} {{ tensor<{block_size}xf32> }}",
            "ptx": f"// block_size={block_size} num_warps={num_warps}\nld.global.f32\n",
        }
        self.metadata = _FakeKernel._Meta(num_warps)


class _FakeJIT:
    """Minimal object that satisfies the bits of JITFunction that Autotuner uses.

    ``run`` emulates a copy kernel: it copies the first ``min(BLOCK_SIZE, N)``
    elements of ``src`` into ``dst``. Configs with ``BLOCK_SIZE < N`` therefore
    produce a WRONG (truncated) result, which is exactly what the correctness
    check should catch. With ``warmup=True`` it returns a ``_FakeKernel`` instead
    of running, mirroring ``run(warmup=True)``.
    """

    def __init__(self, arg_names):
        self.arg_names = arg_names
        self.last_block_size = None

        def _impl():  # base_fn resolution walks .fn until it hits a real function
            return None

        self.fn = _impl

    def run(self, *args, **kwargs):
        block_size = kwargs["BLOCK_SIZE"]
        num_warps = kwargs.get("num_warps", 4)
        self.last_block_size = block_size
        if kwargs.get("warmup", False):
            return _FakeKernel(block_size, num_warps)
        named = dict(zip(self.arg_names, args))
        named.update({k: v for k, v in kwargs.items() if k in self.arg_names})
        dst, src, n = named["dst"], named["src"], named["N"]
        k = min(block_size, n)
        dst.zero_()
        dst[:k] = src[:k]
        return None


def _make_tuner(configs, *, correctness_fn=None, correctness_prune=True, prune_configs_by=None):
    fake = _FakeJIT(["dst", "src", "N", "BLOCK_SIZE"])

    # do_bench gets only kernel_call; recover a deterministic "time" from the
    # block size the fake just ran (smaller block == faster), so the fastest
    # config is the smallest BLOCK_SIZE unless it is pruned.
    def do_bench(kernel_call, quantiles=None):
        kernel_call()
        t = float(fake.last_block_size)
        return [t, t, t]

    tuner = Autotuner(fake, fake.arg_names, configs, key=["N"], reset_to_zero=None, restore_value=["dst"],
                      prune_configs_by=prune_configs_by, do_bench=do_bench, correctness_fn=correctness_fn,
                      correctness_prune=correctness_prune)
    return tuner, fake


def _bs(tuner):
    return tuner.best_config.kwargs["BLOCK_SIZE"]


def _configs():
    return [Config({"BLOCK_SIZE": bs}) for bs in (256, 512, 1024, 2048)]


def test_correctness_prune_excludes_wrong_configs():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    ref = src.clone()

    def correctness_fn(named):
        return torch.equal(named["dst"], ref)

    tuner, _ = _make_tuner(_configs(), correctness_fn=correctness_fn, correctness_prune=True)
    tuner.run(dst, src, N=N, grid=(1, ))

    # Only BLOCK_SIZE >= N produce a full (correct) copy.
    results = {c.kwargs["BLOCK_SIZE"]: ok for c, ok in tuner.correctness_results.items()}
    assert results == {256: False, 512: False, 1024: True, 2048: True}
    # Fastest *correct* config wins (1024 < 2048), not the globally-fastest 256.
    assert _bs(tuner) == 1024


def test_correctness_record_only_does_not_prune():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    ref = src.clone()

    tuner, _ = _make_tuner(_configs(), correctness_fn=lambda named: torch.equal(named["dst"], ref),
                           correctness_prune=False)
    tuner.run(dst, src, N=N, grid=(1, ))

    # Results still recorded...
    results = {c.kwargs["BLOCK_SIZE"]: ok for c, ok in tuner.correctness_results.items()}
    assert results[256] is False and results[1024] is True
    # ...but nothing is pruned, so the globally-fastest (wrong) config wins.
    assert _bs(tuner) == 256


def test_correctness_prune_all_fail_raises():
    N = 4096  # larger than every BLOCK_SIZE -> every config is wrong
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    ref = src.clone()
    tuner, _ = _make_tuner(_configs(), correctness_fn=lambda named: torch.equal(named["dst"], ref),
                           correctness_prune=True)
    with pytest.raises(AutotunerError, match="correctness check"):
        tuner.run(dst, src, N=N, grid=(1, ))


def test_artifact_prune_filters_on_ttgir():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)

    # Keep only configs whose TTGIR contains the 1024-wide tile (BLOCK_SIZE=1024).
    def artifact_prune(config, asm, metadata):
        assert set(asm) >= {"ttir", "ttgir", "ptx"}  # real artifact dict is available
        return "tensor<1024xf32>" in asm["ttgir"]

    tuner, _ = _make_tuner(_configs(), prune_configs_by={"artifact_config_prune": artifact_prune})
    tuner.run(dst, src, N=N, grid=(1, ))

    dropped = {c.kwargs["BLOCK_SIZE"] for c in tuner.pruned_by_artifact}
    assert dropped == {256, 512, 2048}
    assert _bs(tuner) == 1024


def test_artifact_prune_on_metadata():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    configs = [Config({"BLOCK_SIZE": 256}, num_warps=nw) for nw in (1, 2, 4, 8)]

    tuner, _ = _make_tuner(configs, prune_configs_by={"artifact_config_prune": lambda c, asm, md: md.num_warps <= 2})
    tuner.run(dst, src, N=N, grid=(1, ))

    kept_warps = tuner.best_config.num_warps
    assert kept_warps <= 2


def test_artifact_prune_all_pruned_raises():
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    tuner, _ = _make_tuner(_configs(), prune_configs_by={"artifact_config_prune": lambda c, asm, md: False})
    with pytest.raises(AutotunerError, match="artifact pruning"):
        tuner.run(dst, src, N=N, grid=(1, ))


def test_no_hooks_is_unchanged_behavior():
    """When neither hook is set, the fastest config wins (baseline behavior)."""
    N = 1024
    src = torch.arange(N, dtype=torch.float32)
    dst = torch.empty(N, dtype=torch.float32)
    tuner, _ = _make_tuner(_configs())
    tuner.run(dst, src, N=N, grid=(1, ))
    assert _bs(tuner) == 256  # globally fastest, nothing pruned
    assert tuner.correctness_results == {}
    assert tuner.pruned_by_artifact == {}


# ---------------------------------------------------------------------------
# GPU end-to-end tests (real Triton compile + launch)
# ---------------------------------------------------------------------------
@triton.jit
def _sum_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
    # Single-block partial sum: only correct when BLOCK_SIZE >= N.
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_correctness_prune_partial_sum(device="cuda"):
    N = 1024
    src = torch.randn(N, device=device, dtype=torch.float32)
    out = torch.empty(1, device=device, dtype=torch.float32)
    ref = src.sum()

    def correctness_fn(named):
        # named["dst"] holds this config's output after one run.
        return torch.allclose(named["dst"], ref, atol=1e-3, rtol=1e-3)

    configs = [triton.Config({"BLOCK_SIZE": bs}) for bs in (256, 512, 1024, 2048)]

    @triton.autotune(configs=configs, key=["N"], restore_value=["dst"], correctness_fn=correctness_fn,
                     correctness_prune=True)
    @triton.jit
    def kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs, mask=offs < N, other=0.0)
        tl.store(dst, tl.sum(x, axis=0))

    kernel[(1, )](src, out, N)
    # Winner must be a config that actually sums all N elements.
    assert kernel.best_config.kwargs["BLOCK_SIZE"] >= N
    results = {c.kwargs["BLOCK_SIZE"]: ok for c, ok in kernel.correctness_results.items()}
    assert results[256] is False and results[1024] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")
def test_gpu_artifact_prune_on_real_ir(device="cuda"):
    N = 1024
    src = torch.randn(N, device=device, dtype=torch.float32)
    out = torch.empty(1, device=device, dtype=torch.float32)

    seen = {}

    def artifact_prune(config, asm, metadata):
        # Real compiled artifacts must be present and inspectable.
        assert "ttgir" in asm and "ptx" in asm
        assert isinstance(asm["ttgir"], str) and len(asm["ttgir"]) > 0
        seen[config.num_warps] = ("tt." in asm["ttgir"])
        return metadata.num_warps <= 4  # keep only <=4 warp configs

    configs = [triton.Config({"BLOCK_SIZE": 1024}, num_warps=nw) for nw in (1, 2, 4, 8)]

    @triton.autotune(configs=configs, key=["N"], prune_configs_by={"artifact_config_prune": artifact_prune})
    @triton.jit
    def kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs, mask=offs < N, other=0.0)
        tl.store(dst, tl.sum(x, axis=0))

    kernel[(1, )](src, out, N)
    assert kernel.best_config.num_warps <= 4
    dropped = {c.num_warps for c in kernel.pruned_by_artifact}
    assert 8 in dropped
    assert all(seen.values())  # every inspected TTGIR really contained IR text
