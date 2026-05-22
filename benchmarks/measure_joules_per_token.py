#!/usr/bin/env python3
"""
Measure joules-per-token for LLM inference using pynvml GPU energy counters.

This script measures GPU energy consumption during transformer inference,
separating prompt processing (prefill) from token generation (decode) phases.
"""

import sys
import time
from typing import Dict, Tuple, Optional

try:
    import pynvml
except ImportError:
    print("ERROR: pynvml not installed. Run: pip install pynvml")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Run: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers")
    sys.exit(1)


class GPUEnergyMonitor:
    """Monitor GPU energy consumption using NVIDIA pynvml."""

    def __init__(self):
        """Initialize pynvml and check for NVIDIA GPU."""
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            print(f"ERROR: Failed to initialize pynvml: {e}")
            print("This script requires an NVIDIA GPU with energy counter support.")
            sys.exit(1)

        self.device_count = pynvml.nvmlDeviceGetCount()
        if self.device_count == 0:
            print("ERROR: No NVIDIA GPUs detected")
            sys.exit(1)

        # Use first GPU (device 0)
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._validate_energy_support()

    def _validate_energy_support(self):
        """Check if GPU supports energy counters."""
        try:
            # Try reading energy to verify support
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except pynvml.NVMLError:
            print("ERROR: GPU does not support energy counters")
            print("Most modern NVIDIA GPUs support this feature.")
            sys.exit(1)

    def get_gpu_info(self) -> Dict[str, str]:
        """Get GPU model, driver, and CUDA version."""
        name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode('utf-8')

        cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda_major = cuda_version // 1000
        cuda_minor = (cuda_version % 1000) // 10

        return {
            "gpu_model": name,
            "driver_version": driver_version,
            "cuda_version": f"{cuda_major}.{cuda_minor}"
        }

    def get_energy_millijoules(self) -> int:
        """Read current total energy consumption in millijoules."""
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def measure_energy_delta(self, func) -> Tuple[float, any]:
        """
        Measure energy consumed by executing func.

        Returns:
            (energy_joules, func_result)
        """
        energy_start = self.get_energy_millijoules()
        result = func()
        energy_end = self.get_energy_millijoules()

        energy_millijoules = energy_end - energy_start
        energy_joules = energy_millijoules / 1000.0

        return energy_joules, result

    def cleanup(self):
        """Shutdown pynvml."""
        pynvml.nvmlShutdown()


def check_gpu_load():
    """
    Check if GPU is under load from background processes.
    Warns user if GPU utilization > 10%.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    pynvml.nvmlShutdown()

    if util.gpu > 10:
        print(f"WARNING: GPU utilization is {util.gpu}% before benchmark")
        print("Background processes may affect measurement accuracy.")
        print("Consider closing other GPU applications.")
        input("Press Enter to continue or Ctrl+C to abort...")


if __name__ == "__main__":
    # Basic smoke test
    print("Testing GPU energy monitoring...")
    monitor = GPUEnergyMonitor()
    info = monitor.get_gpu_info()
    print(f"GPU: {info['gpu_model']}")
    print(f"Driver: {info['driver_version']}")
    print(f"CUDA: {info['cuda_version']}")

    # Test energy measurement
    def sleep_test():
        time.sleep(0.1)
        return "test"

    energy, result = monitor.measure_energy_delta(sleep_test)
    print(f"Energy delta test: {energy:.3f} J")

    monitor.cleanup()
    print("GPU energy monitoring OK")
