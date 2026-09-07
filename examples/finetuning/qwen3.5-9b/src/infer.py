from __future__ import annotations

import argparse
import time

import torch

from .data import load_jsonl, write_jsonl
from .project import DEFAULT_MODEL, configure_local_caches, inside_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible predictions.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    configure_local_caches()
    data_path = inside_project(args.data, must_exist=True)
    output_path = inside_project(args.output)
    model_path = inside_project(args.model, must_exist=True)
    adapter_path = inside_project(args.adapter, must_exist=True) if args.adapter else None

    from peft import PeftModel

    from .modeling import load_processor, load_quantized_model, require_cuda_free

    examples = load_jsonl(data_path)
    require_cuda_free(12.0, "inference")
    processor, tokenizer = load_processor(str(model_path))
    model = load_quantized_model(str(model_path))
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()

    results = []
    for index, example in enumerate(examples, start=1):
        encoded = processor.apply_chat_template(
            example.prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        new_tokens = generated[0, prompt_length:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(
            {
                "id": example.id,
                "reference": example.reference,
                "prediction": prediction,
                "generated_tokens": int(new_tokens.numel()),
                "elapsed_seconds": round(elapsed, 4),
                "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "adapter": str(adapter_path) if adapter_path else None,
            }
        )
        print(f"[{index}/{len(examples)}] {example.id}: {elapsed:.2f}s")

    write_jsonl(output_path, results)
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
