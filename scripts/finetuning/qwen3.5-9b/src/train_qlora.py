from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass

import torch

from .data import Example, load_jsonl
from .project import DEFAULT_MODEL, ROOT, configure_local_caches, inside_project


class ConversationDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[Example], processor, max_length: int):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        full = self.processor.apply_chat_template(
            example.messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            enable_thinking=False,
        )
        prompt = self.processor.apply_chat_template(
            example.prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
        )
        # Multimodal processors return a batch dimension even for one text-only
        # sample when return_tensors is omitted. Normalize it to a flat sequence.
        full_ids = full["input_ids"]
        if full_ids and isinstance(full_ids[0], list):
            full_ids = full_ids[0]
        prompt_ids = prompt["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        full_mask = full.get("attention_mask")
        if full_mask and isinstance(full_mask[0], list):
            full_mask = full_mask[0]
        input_ids = list(full_ids[: self.max_length])
        attention_mask = list((full_mask or [1] * len(input_ids))[: self.max_length])
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError(
                f"Example {example.id!r} has no assistant tokens within max_length={self.max_length}"
            )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class CompletionCollator:
    tokenizer: object

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only 4-bit QLoRA for Qwen3.5-9B.")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "qlora"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epochs when positive; useful for a one-step smoke test.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Checkpoint directory inside the project from which to resume Trainer state.",
    )
    args = parser.parse_args()

    configure_local_caches()
    train_path = inside_project(args.train_data, must_exist=True)
    eval_path = inside_project(args.eval_data, must_exist=True) if args.eval_data else None
    model_path = inside_project(args.model, must_exist=True)
    output_path = inside_project(args.output)
    resume_path = (
        inside_project(args.resume_from_checkpoint, must_exist=True)
        if args.resume_from_checkpoint
        else None
    )
    output_path.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    from .modeling import load_processor, load_quantized_model, require_cuda_free

    set_seed(args.seed)
    require_cuda_free(24.0, "QLoRA training")
    processor, tokenizer = load_processor(str(model_path))
    model = load_quantized_model(str(model_path))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen3.5 mixes full attention and Gated DeltaNet layers. Naming the
        # language projections explicitly avoids allocating adapters in the
        # frozen vision encoder during text-only SFT.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_dataset = ConversationDataset(load_jsonl(train_path), processor, args.max_length)
    eval_dataset = (
        ConversationDataset(load_jsonl(eval_path), processor, args.max_length) if eval_path else None
    )

    training_kwargs = dict(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        optim="paged_adamw_8bit",
    )
    argument_parameters = inspect.signature(TrainingArguments).parameters
    if "warmup_ratio" in argument_parameters:
        training_kwargs["warmup_ratio"] = 0.03
    else:
        # Transformers 5 accepts a fraction in warmup_steps.
        training_kwargs["warmup_steps"] = 0.03
    if eval_dataset:
        strategy_name = (
            "eval_strategy"
            if "eval_strategy" in argument_parameters
            else "evaluation_strategy"
        )
        training_kwargs[strategy_name] = "steps"
        training_kwargs["eval_steps"] = args.eval_steps
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    final_path = output_path / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"Saved final adapter to {final_path}")


if __name__ == "__main__":
    main()
