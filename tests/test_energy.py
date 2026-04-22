"""Unit tests for sparsefire.energy. No GPU required — nvml is fully mocked."""

from __future__ import annotations

import sys
import time
import types

import pytest

from sparsefire.energy import EnergyMeter, EnergyResult, bootstrap_ci


class FakeNvml:
    """In-memory stand-in for nvidia-ml-py."""

    def __init__(self, start_mj: int = 1_000_000, delta_mj: int = 12_000, power_mw: int = 300_000):
        self.start = start_mj
        self.delta = delta_mj
        self.power_mw = power_mw
        self._calls = 0

    # API surface used by EnergyMeter
    def nvmlInit(self): ...
    def nvmlDeviceGetHandleByIndex(self, i: int):
        return "fake-handle"

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        self._calls += 1
        return self.start if self._calls == 1 else self.start + self.delta

    def nvmlDeviceGetPowerUsage(self, handle):
        return self.power_mw


@pytest.fixture
def fake_nvml(monkeypatch):
    mod = types.ModuleType("pynvml")
    fake = FakeNvml()
    mod.nvmlInit = fake.nvmlInit
    mod.nvmlDeviceGetHandleByIndex = fake.nvmlDeviceGetHandleByIndex
    mod.nvmlDeviceGetTotalEnergyConsumption = fake.nvmlDeviceGetTotalEnergyConsumption
    mod.nvmlDeviceGetPowerUsage = fake.nvmlDeviceGetPowerUsage
    monkeypatch.setitem(sys.modules, "pynvml", mod)
    return fake


def test_energy_meter_computes_delta(fake_nvml):
    with EnergyMeter(device_index=0, sample_interval_ms=10) as m:
        time.sleep(0.05)  # let the sampler grab a few points
    r = m.result
    assert isinstance(r, EnergyResult)
    # 12_000 mJ delta = 12 J
    assert r.total_energy_j == pytest.approx(12.0)
    assert r.wallclock_s > 0
    assert r.mean_power_w == pytest.approx(300.0, rel=0.01)


def test_energy_result_per_token(fake_nvml):
    with EnergyMeter(sample_interval_ms=10) as m:
        time.sleep(0.02)
    assert m.result.per_token(256) == pytest.approx(12.0 / 256)


def test_energy_result_per_token_rejects_zero():
    r = EnergyResult(total_energy_j=1.0, wallclock_s=1.0, mean_power_w=1.0, peak_power_w=1.0)
    with pytest.raises(ValueError):
        r.per_token(0)


def test_bootstrap_ci_on_constant_input():
    mean, lo, hi = bootstrap_ci([5.0] * 100, n_bootstrap=500, seed=0)
    assert mean == 5.0
    assert lo == 5.0 and hi == 5.0


def test_bootstrap_ci_reasonable_width():
    import random

    rng = random.Random(42)
    vals = [rng.gauss(10.0, 1.0) for _ in range(100)]
    mean, lo, hi = bootstrap_ci(vals, n_bootstrap=2000, seed=0)
    assert lo < mean < hi
    # 95% CI on n=100 from sigma=1 should be ~0.2 wide on each side
    assert (hi - lo) < 0.6


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([], n_bootstrap=10)


def test_meter_result_before_exit_errors(fake_nvml):
    m = EnergyMeter()
    with pytest.raises(RuntimeError):
        _ = m.result
