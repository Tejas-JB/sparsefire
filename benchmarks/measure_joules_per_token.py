#!/usr/bin/env python3
"""
Measure joules-per-token for LLM inference using pynvml GPU energy counters.

This script measures GPU energy consumption during transformer inference,
separating prompt processing (prefill) from token generation (decode) phases.
"""

import sys
import time
from typing import Dict, Tuple, Optional, Callable, Any

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

        # Store device count for potential multi-GPU support in future
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

    def measure_energy_delta(self, func: Callable[[], Any]) -> Tuple[float, Any]:
        """
        Measure energy consumed by executing func.

        Returns:
            (energy_joules, func_result)
        """
        energy_start = self.get_energy_millijoules()
        try:
            result = func()
        finally:
            energy_end = self.get_energy_millijoules()

        energy_millijoules = energy_end - energy_start
        energy_joules = energy_millijoules / 1000.0

        return energy_joules, result

    def cleanup(self):
        """Shutdown pynvml."""
        pynvml.nvmlShutdown()


def check_gpu_load(monitor: GPUEnergyMonitor):
    """
    Check if GPU is under load from background processes.
    Warns user if GPU utilization > 10%.
    """
    util = pynvml.nvmlDeviceGetUtilizationRates(monitor.handle)

    if util.gpu > 10:
        print(f"WARNING: GPU utilization is {util.gpu}% before benchmark")
        print("Background processes may affect measurement accuracy.")
        print("Consider closing other GPU applications.")
        input("Press Enter to continue or Ctrl+C to abort...")


class LLMBenchmark:
    """Benchmark LLM inference energy consumption."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        """
        Initialize model and tokenizer.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            print("ERROR: CUDA not available. This benchmark requires GPU.")
            sys.exit(1)

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        # Set pad token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded successfully")

    def warmup(self, num_tokens: int = 10):
        """
        Warmup run to initialize GPU state.

        Args:
            num_tokens: Number of tokens to generate
        """
        print("Running warmup...")
        prompt = "Warmup prompt"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                temperature=0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Warmup complete")

    def measure_prompt_processing(
        self,
        prompt: str,
        monitor: GPUEnergyMonitor
    ) -> Tuple[float, torch.Tensor]:
        """
        Measure energy for prompt processing (prefill phase).

        Returns:
            (energy_joules, prompt_input_ids)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs['input_ids'].shape[1]

        def process_prompt():
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs

        energy, outputs = monitor.measure_energy_delta(process_prompt)

        return energy, inputs['input_ids']

    def measure_generation(
        self,
        prompt_input_ids: torch.Tensor,
        num_tokens: int,
        monitor: GPUEnergyMonitor
    ) -> Tuple[float, int]:
        """
        Measure energy for token generation (decode phase).

        Args:
            prompt_input_ids: Tokenized prompt from prefill phase
            num_tokens: Number of tokens to generate
            monitor: Energy monitor

        Returns:
            (energy_joules, actual_tokens_generated)
        """
        def generate():
            with torch.no_grad():
                output_ids = self.model.generate(
                    prompt_input_ids,
                    max_new_tokens=num_tokens,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            return output_ids

        energy, output_ids = monitor.measure_energy_delta(generate)

        # Calculate actual tokens generated
        generated_tokens = output_ids.shape[1] - prompt_input_ids.shape[1]

        return energy, generated_tokens


def run_single_measurement(
    benchmark: LLMBenchmark,
    monitor: GPUEnergyMonitor,
    prompt: str,
    num_tokens: int
) -> Dict[str, float]:
    """
    Run single measurement of prompt + generation.

    Returns:
        Dict with energy metrics
    """
    # Measure prompt processing
    prompt_energy, prompt_ids = benchmark.measure_prompt_processing(prompt, monitor)
    prompt_token_count = prompt_ids.shape[1]

    # Clear cache between phases
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure generation
    gen_energy, gen_token_count = benchmark.measure_generation(
        prompt_ids, num_tokens, monitor
    )

    total_energy = prompt_energy + gen_energy
    total_tokens = prompt_token_count + gen_token_count

    return {
        "prompt_tokens": prompt_token_count,
        "generated_tokens": gen_token_count,
        "total_tokens": total_tokens,
        "prompt_phase_joules": prompt_energy,
        "generation_phase_joules": gen_energy,
        "total_energy_joules": total_energy,
        "joules_per_token_overall": total_energy / total_tokens if total_tokens > 0 else 0,
        "joules_per_token_generation": gen_energy / gen_token_count if gen_token_count > 0 else 0
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure joules-per-token for LLM inference"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run basic test mode"
    )
    args = parser.parse_args()

    if args.test:
        print("=== GPU Energy Monitoring Test ===")
        monitor = GPUEnergyMonitor()
        info = monitor.get_gpu_info()
        print(f"GPU: {info['gpu_model']}")
        print(f"Driver: {info['driver_version']}")
        print(f"CUDA: {info['cuda_version']}")

        def sleep_test():
            time.sleep(0.1)
            return "test"

        energy, result = monitor.measure_energy_delta(sleep_test)
        print(f"Energy delta test: {energy:.3f} J")
        monitor.cleanup()
        print("\n=== LLM Benchmark Test ===")

        # Test inference measurement
        monitor = GPUEnergyMonitor()
        benchmark = LLMBenchmark()
        check_gpu_load(monitor)
        benchmark.warmup(num_tokens=5)

        result = run_single_measurement(
            benchmark,
            monitor,
            prompt="The quick brown fox",
            num_tokens=10
        )

        print(f"\nMeasurement result:")
        print(f"  Prompt tokens: {result['prompt_tokens']}")
        print(f"  Generated tokens: {result['generated_tokens']}")
        print(f"  Total energy: {result['total_energy_joules']:.3f} J")
        print(f"  J/token (overall): {result['joules_per_token_overall']:.4f}")
        print(f"  J/token (generation): {result['joules_per_token_generation']:.4f}")

        monitor.cleanup()
        print("\nAll tests passed!")
    else:
        print("Use --test flag to run basic tests")
        print("Full benchmark mode coming in next task...")
