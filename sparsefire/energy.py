"""Energy measurement via NVML.

Primary energy = nvmlDeviceGetTotalEnergyConsumption delta before/after generation.
Diagnostic power trace = nvmlDeviceGetPowerUsage polled at sample_interval_ms.

This module MUST import without CUDA/nvml present (Mac dev) — nvml calls are
lazy-loaded inside EnergyMeter so import is safe; tests mock nvml directly.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class EnergyResult:
    total_energy_j: float
    wallclock_s: float
    mean_power_w: float
    peak_power_w: float
    power_trace_w: list[float] = field(default_factory=list)
    sample_interval_ms: int = 50

    @property
    def joules_per_token(self) -> float:
        raise NotImplementedError("set externally via EnergyResult.per_token(n_tokens)")

    def per_token(self, n_tokens: int) -> float:
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        return self.total_energy_j / n_tokens


class EnergyMeter:
    """Context manager wrapping a generation call.

    Usage:
        with EnergyMeter(device_index=0) as m:
            model.generate(...)
        result: EnergyResult = m.result
    """

    def __init__(self, device_index: int = 0, sample_interval_ms: int = 50) -> None:
        self.device_index = device_index
        self.sample_interval_ms = sample_interval_ms
        self._start_energy_mj: int | None = None
        self._start_time: float | None = None
        self._power_samples_mw: list[int] = []
        self._sampler: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._handle = None
        self._pynvml = None
        self._result: EnergyResult | None = None

    def _lazy_init_nvml(self):
        if self._pynvml is not None:
            return
        import pynvml  # nvidia-ml-py provides this

        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def __enter__(self) -> EnergyMeter:
        self._lazy_init_nvml()
        self._start_energy_mj = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        self._start_time = time.perf_counter()
        self._stop_event.clear()
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()
        return self

    def _sample_loop(self) -> None:
        interval = self.sample_interval_ms / 1000.0
        while not self._stop_event.is_set():
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._power_samples_mw.append(mw)
            except Exception:  # noqa: BLE001 — telemetry hiccup shouldn't kill a run
                pass
            time.sleep(interval)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)
        end_energy_mj = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        wallclock_s = time.perf_counter() - (self._start_time or 0.0)
        total_energy_j = (end_energy_mj - (self._start_energy_mj or 0)) / 1000.0
        trace_w = [mw / 1000.0 for mw in self._power_samples_mw]
        mean_power_w = sum(trace_w) / len(trace_w) if trace_w else 0.0
        peak_power_w = max(trace_w) if trace_w else 0.0
        self._result = EnergyResult(
            total_energy_j=total_energy_j,
            wallclock_s=wallclock_s,
            mean_power_w=mean_power_w,
            peak_power_w=peak_power_w,
            power_trace_w=trace_w,
            sample_interval_ms=self.sample_interval_ms,
        )

    @property
    def result(self) -> EnergyResult:
        if self._result is None:
            raise RuntimeError("EnergyMeter.result accessed before context exit")
        return self._result


def bootstrap_ci(
    values: list[float], n_bootstrap: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    import random

    if not values:
        raise ValueError("values must be non-empty")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_bootstrap * (alpha / 2))]
    hi = means[int(n_bootstrap * (1 - alpha / 2))]
    mean = sum(values) / n
    return mean, lo, hi


def lock_gpu_clocks(freq_mhz: int, device_index: int = 0) -> None:
    """Pin GPU clocks. Requires root/privileged container."""
    subprocess.run(
        ["nvidia-smi", "-i", str(device_index), "-lgc", f"{freq_mhz},{freq_mhz}"],
        check=True,
    )


def unlock_gpu_clocks(device_index: int = 0) -> None:
    subprocess.run(["nvidia-smi", "-i", str(device_index), "-rgc"], check=True)


@contextlib.contextmanager
def locked_clocks(freq_mhz: int, device_index: int = 0, enable: bool = True):
    if not enable:
        yield
        return
    lock_gpu_clocks(freq_mhz, device_index)
    try:
        yield
    finally:
        with contextlib.suppress(subprocess.CalledProcessError):
            unlock_gpu_clocks(device_index)


def warmup(generate_fn, seconds: int = 60) -> None:
    """Call generate_fn() in a loop until `seconds` have elapsed."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        generate_fn()
