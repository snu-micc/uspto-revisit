from __future__ import annotations

import time

import torch


def require_cuda_free(minimum_gib: float, operation: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for {operation}, but no CUDA device is visible")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"Refusing to start {operation}: selected GPU has {free_gib:.1f}/{total_gib:.1f} GiB "
            f"free, below the {minimum_gib:.1f} GiB safety threshold"
        )
    print(f"GPU safety check for {operation}: {free_gib:.1f}/{total_gib:.1f} GiB free")


def wait_for_cuda_free(
    minimum_gib: float,
    operation: str,
    timeout_minutes: float,
    poll_seconds: float = 60.0,
) -> None:
    if timeout_minutes <= 0:
        require_cuda_free(minimum_gib, operation)
        return
    started = time.monotonic()
    timeout_seconds = timeout_minutes * 60
    while True:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is required for {operation}, but no CUDA device is visible"
            )
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gib = free_bytes / 1024**3
        total_gib = total_bytes / 1024**3
        if free_gib >= minimum_gib:
            print(
                f"GPU safety check for {operation}: "
                f"{free_gib:.1f}/{total_gib:.1f} GiB free"
            )
            return
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"Timed out waiting for {operation}: selected GPU has "
                f"{free_gib:.1f}/{total_gib:.1f} GiB free, below the "
                f"{minimum_gib:.1f} GiB safety threshold"
            )
        remaining = timeout_seconds - elapsed
        delay = min(poll_seconds, remaining)
        print(
            f"Waiting for GPU memory for {operation}: "
            f"{free_gib:.1f}/{total_gib:.1f} GiB free; "
            f"need {minimum_gib:.1f} GiB (next check in {delay:.0f}s)"
        )
        time.sleep(delay)


def quantization_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_processor(model_path: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor, tokenizer


def load_quantized_model(model_path: str):
    from transformers import AutoModelForMultimodalLM

    return AutoModelForMultimodalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config(),
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
