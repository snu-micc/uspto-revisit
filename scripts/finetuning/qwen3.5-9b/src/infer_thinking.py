from __future__ import annotations

import argparse
import time

import torch

from .data import load_jsonl, write_jsonl
from .project import DEFAULT_MODEL, configure_local_caches, inside_project


EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited thinking budget, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)


def as_eos_list(eos_token_id: int | list[int] | None) -> list[int]:
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, list):
        return eos_token_id
    return [eos_token_id]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate answers with a bounded Qwen thinking phase."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--adapter")
    parser.add_argument("--thinking-level", choices=("off", "low", "medium", "high"), required=True)
    parser.add_argument("--thinking-budget", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cuda-wait-minutes", type=float, default=0)
    args = parser.parse_args()

    if args.thinking_level == "off" and args.thinking_budget != 0:
        raise ValueError("off requires --thinking-budget 0")
    if args.thinking_level != "off" and args.thinking_budget <= 0:
        raise ValueError("thinking levels require a positive --thinking-budget")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.cuda_wait_minutes < 0:
        raise ValueError("--cuda-wait-minutes cannot be negative")

    configure_local_caches()
    data_path = inside_project(args.data, must_exist=True)
    output_path = inside_project(args.output)
    model_path = inside_project(args.model, must_exist=True)
    adapter_path = inside_project(args.adapter, must_exist=True) if args.adapter else None

    from peft import PeftModel

    from .modeling import load_processor, load_quantized_model, wait_for_cuda_free

    examples = load_jsonl(data_path)
    if args.limit is not None:
        examples = examples[: args.limit]
    wait_for_cuda_free(
        9.0, "thinking inference", timeout_minutes=args.cuda_wait_minutes
    )
    processor, tokenizer = load_processor(str(model_path))
    model = load_quantized_model(str(model_path))
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()

    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_id, int) or think_end_id < 0:
        raise ValueError("tokenizer does not define a </think> token")
    model_eos_ids = as_eos_list(tokenizer.eos_token_id)
    phase_one_eos = list(dict.fromkeys(model_eos_ids + [think_end_id]))

    results = []
    for index, example in enumerate(examples, start=1):
        enable_thinking = args.thinking_level != "off"
        encoded = processor.apply_chat_template(
            example.prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        thinking = ""
        thinking_tokens = 0
        budget_reached = False
        if not enable_thinking:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            answer_ids = generated[0, prompt_length:]
            total_generated_tokens = int(answer_ids.numel())
        else:
            with torch.inference_mode():
                phase_one = model.generate(
                    **encoded,
                    max_new_tokens=args.thinking_budget,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=phase_one_eos,
                )
            phase_one_ids = phase_one[0, prompt_length:]
            phase_one_list = phase_one_ids.tolist()
            thinking_tokens = len(phase_one_list)
            ended_thinking = think_end_id in phase_one_list
            ended_response = any(token_id in phase_one_list for token_id in model_eos_ids)

            if ended_thinking:
                think_end_position = phase_one_list.index(think_end_id)
                reasoning_ids = phase_one_ids[:think_end_position]
                continuation_text = "\n\n"
            else:
                reasoning_ids = phase_one_ids
                continuation_text = EARLY_STOPPING_TEXT
                budget_reached = not ended_response

            thinking = tokenizer.decode(
                reasoning_ids, skip_special_tokens=True
            ).strip()

            if ended_response and not ended_thinking:
                answer_ids = phase_one_ids
                total_generated_tokens = int(phase_one_ids.numel())
            else:
                continuation_ids = tokenizer(
                    continuation_text,
                    return_tensors="pt",
                    return_attention_mask=False,
                    add_special_tokens=False,
                ).input_ids.to(model.device)
                continued_input_ids = torch.cat([phase_one, continuation_ids], dim=-1)
                attention_mask = torch.ones_like(continued_input_ids, dtype=torch.long)
                continuation_length = continued_input_ids.shape[-1]
                with torch.inference_mode():
                    phase_two = model.generate(
                        input_ids=continued_input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                answer_ids = phase_two[0, continuation_length:]
                total_generated_tokens = int(
                    phase_one_ids.numel() + answer_ids.numel()
                )

        elapsed = time.perf_counter() - started
        prediction = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        results.append(
            {
                "id": example.id,
                "reference": example.reference,
                "prediction": prediction,
                "thinking": thinking,
                "thinking_level": args.thinking_level,
                "thinking_budget": args.thinking_budget,
                "thinking_tokens": thinking_tokens,
                "answer_tokens": int(answer_ids.numel()),
                "generated_tokens": total_generated_tokens,
                "budget_reached": budget_reached,
                "elapsed_seconds": round(elapsed, 4),
                "peak_gpu_mib": round(
                    torch.cuda.max_memory_allocated() / 1024**2, 1
                ),
                "adapter": str(adapter_path) if adapter_path else None,
            }
        )
        print(
            f"[{index}/{len(examples)}] {example.id}: level={args.thinking_level} "
            f"think={thinking_tokens} answer={int(answer_ids.numel())} "
            f"budget_reached={budget_reached} elapsed={elapsed:.2f}s"
        )

    write_jsonl(output_path, results)
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
