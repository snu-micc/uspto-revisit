# Fine-tuning examples

This directory contains the 70 task-specific examples used for fine-tuning.
They do not overlap with the 400-paragraph evaluation benchmark.

## gpt-4.1-mini

`openai/samples_for_finetuning_final.jsonl` uses the chat-completions JSONL
format with one system, user, and assistant message per record. The assistant
target is the structured reaction-summary JSON used by the extraction
pipeline.

## qwen-3.5-9B

`qwen3.5-9b/qwen3_5_9b_sft_messages.jsonl` contains the same examples in a
Hugging Face/TRL-compatible `messages` schema. The accompanying converter
removes double JSON encoding, validates every target, and preserves all 70
records. Apply the tokenizer revision's own chat template during preprocessing;
do not insert model-specific special tokens directly into the JSONL file.

The Qwen run used **4-bit QLoRA**, with 56 training examples and 14 held-out
examples from these 70 records. The held-out examples were used for checkpoint
selection and baseline/fine-tuned JSON-format, ROUGE-L, and exact-match
comparison; they are separate from the 400-paragraph publication benchmark.

The reported base-model revision, training settings, checkpoint losses, and
hardware/software environment are recorded in
[qwen3.5-9b/README.md](qwen3.5-9b/README.md) and the machine-readable
[reported_run_config.json](qwen3.5-9b/reported_run_config.json).
