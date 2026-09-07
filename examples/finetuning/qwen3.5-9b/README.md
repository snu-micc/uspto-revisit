# Qwen3.5-9B fine-tuning data

`qwen3_5_9b_sft_messages.jsonl` contains the existing GPT fine-tuning examples
in the conversational format expected by Hugging Face datasets and TRL:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"{...}"}]}
```

The source assistant targets were JSON objects encoded inside an additional
JSON string. The converter removes that extra encoding so that every assistant
target is exactly one valid JSON object and contains no Markdown or reasoning
text.

For Qwen3.5-9B, render each conversation with the model's own chat template and
disable thinking for this extraction task:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3.5-9B",
    revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
)
text = processor.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,
)
```

Do not manually insert Qwen special tokens into the JSONL file. Applying the
model's chat template during preprocessing keeps the training format tied to
the exact tokenizer revision used for the experiment.

The available source file contains 70 records, not 75. The converter preserves
all 70 records without inventing or deleting examples.

## Reported training workflow

The study used **QLoRA, not full fine-tuning**. The base weights were loaded
in 4-bit NF4 format and only the LoRA parameters were trained. The workflow
was:

1. Convert the existing GPT training data into Qwen `messages` JSONL.
2. Split the 70 examples into 56 training and 14 evaluation examples.
3. Generate baseline outputs from the original `Qwen/Qwen3.5-9B` model on
   those 14 evaluation examples.
4. Train using 4-bit QLoRA, with loss computed only on assistant responses.
5. Select `checkpoint-42` (epoch 3), which had the lowest evaluation loss.
6. Generate fine-tuned outputs on the same 14 evaluation examples using the
   selected checkpoint.
7. Compare JSON format, ROUGE-L, and exact match between the baseline and
   fine-tuned outputs.

These 14 examples were also used for checkpoint selection; this comparison is
a held-out validation analysis, not the independent 400-paragraph publication
benchmark. The 70 examples do not overlap with that benchmark. Training used
`enable_thinking=False`; the none/low/high configurations released for the
400-paragraph benchmark are separate inference-time evaluations.

## Reported configuration

The following values document the reported training run. A structured copy is
provided in [reported_run_config.json](reported_run_config.json).

| Setting | Reported value |
| --- | --- |
| Fine-tuning method | QLoRA (not full fine-tuning) |
| Base model | `Qwen/Qwen3.5-9B` |
| Exact base-model revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Quantization | 4-bit NF4 |
| Double quantization | Enabled |
| Quantization compute dtype | `bfloat16` |
| LoRA rank | `r=16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| LoRA bias | `none` |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters / total parameters | 43,278,336 / 9,453,092,080 |
| Trainable fraction | Approximately 0.4578% |
| Epochs | 5 |
| Total optimizer steps | 70 |
| Learning rate | `2e-4` |
| Learning-rate scheduler | `cosine` |
| Warmup ratio | `0.03` |
| Optimizer | `paged_adamw_8bit` |
| Per-device training batch size | 1 |
| Per-device evaluation batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Maximum sequence length | 2,048 tokens |
| Random seed | 42 |
| Data seed | 42 |
| Training precision | BF16 |
| TF32 | Enabled |
| Gradient checkpointing | Enabled |
| Thinking mode | `enable_thinking=False` |
| Loss scope | System/user labels masked with `-100`; assistant responses only |
| Training examples | 56 |
| Evaluation examples | 14 |
| Evaluation/save interval | Every 14 optimizer steps |
| Best-model metric | `eval_loss` (lower is better) |
| Best checkpoint | `checkpoint-42`, epoch 3 |
| Best evaluation loss | 0.0712597 |
| Final-epoch evaluation loss | 0.0799484 |
| Training resume point | `checkpoint-14` |
| GPU | NVIDIA RTX 6000 Ada Generation |
| GPU memory | 49,140 MiB (approximately 48 GB) |
| Slurm CPU/RAM allocation | 8 CPU cores / 24 GB RAM |
| Python | `3.11.15` |
| PyTorch / CUDA | `2.11.0+cu128` / CUDA 12.8 |
| Transformers | `5.15.0.dev0` |
| PEFT | `0.20.0` |
| bitsandbytes | `0.50.0` |

With 56 training examples, batch size 1, and gradient accumulation 4, each
epoch has 14 optimizer steps. Five epochs give 70 steps, and step 42 is the
end of epoch 3. The best checkpoint was chosen by evaluation loss, not simply
by taking the final epoch.

## Released artifacts and reproduction scope

The repository releases the source GPT examples, converted Qwen examples,
format-conversion code, original training/inference/scoring modules in `src/`,
the exact `data/train.jsonl` and `data/eval.jsonl` splits, and the original
14-example baseline/fine-tuned predictions under `outputs/`. The best adapter
is available as a [GitHub Release](https://github.com/snu-micc/uspto-revisit/releases/tag/qwen3.5-9b-qlora-checkpoint-42).
The processed outputs for the separate 400-paragraph benchmark remain in the
repository's `result/model_outputs/` directory.

Run the converter from the repository root:

```bash
python examples/finetuning/qwen3.5-9b/convert_gpt_examples_to_qwen35.py --input examples/finetuning/openai/samples_for_finetuning_final.jsonl --output examples/finetuning/qwen3.5-9b/qwen3_5_9b_sft_messages.jsonl --summary conversion_summary.json
```

`src/train_qlora.py` is unchanged from the supplied training script. Its default
epochs/accumulation/intervals differ from the reported run, so use the explicit
arguments below. `src/project.py` now resolves paths relative to this directory
instead of the author's server path. The Slurm example is adapted to a generic
one-GPU request and the submission directory; choose a partition suitable for
your cluster. The historical run resumed from `checkpoint-14`; the launch
example supports a fresh run or an explicit `QWEN_RESUME` path.

`reported_run_config.json` is a machine-readable experiment record, not a
directly executable `Trainer` configuration. [source_manifest.json](source_manifest.json)
records hashes of the released data and modules. The 56/14 split was verified
against the original seed-42 shuffle, and the full five-epoch evaluation losses
are recorded in [evaluation_loss_history.json](evaluation_loss_history.json).

The exact commit corresponding to `Transformers 5.15.0.dev0` is not recorded
in the supplied environment (`direct_url.json` was empty). The original baseline
launch command was also not preserved: its outputs contain up to 1,014 generated
tokens, while the fine-tuned Slurm command explicitly used a 1,024-token cap.
The commands below explicitly set 1,024 for both models. These remaining details
limit bit-for-bit training/generation reproduction; the saved-output metrics and
the exact split are reproducible without a GPU. The JSON record uses `null` for
unknown historical settings instead of assuming values.

The reported QLoRA environment is separate from the repository's
evaluation/LocalMapper environment. Its package versions describe the
historical run; they are not installation requirements for recalculating
Table 2 or Table 3.

## Reproduce the split and saved-output comparison

From the repository root, change into this Qwen directory:

```bash
cd examples/finetuning/qwen3.5-9b
python -m src.validate_data --train data/train.jsonl --eval data/eval.jsonl
python -m src.prepare_split --source qwen3_5_9b_sft_messages.jsonl --train-output tmp/train.jsonl --eval-output tmp/eval.jsonl --eval-fraction 0.2 --seed 42
python -m src.compare --baseline outputs/baseline.jsonl --finetuned outputs/finetuned.jsonl --output outputs/comparison.md
```

The regenerated split preserves source order within each partition and matches
the committed `data/train.jsonl` and `data/eval.jsonl`. The original 14-example
outputs give:

| Metric | Baseline | Fine-tuned |
| --- | ---: | ---: |
| Normalized text exact match | 0.0000 | 0.0714 |
| ROUGE-L F1 | 0.2300 | 0.7721 |
| JSON valid | 1.0000 | 1.0000 |
| Schema valid | 0.7143 | 1.0000 |
| JSON exact match | 0.0000 | 0.0714 |

`src/compare.py` specifies the metrics exactly. Text exact match casefolds and
normalizes whitespace. ROUGE-L uses longest common subsequence over those
whitespace-separated tokens, with per-example F1 averaged across examples.
JSON exact match compares parsed JSON values. Schema validity requires exactly
`Reactants, Solvents, Catalysts`, `Products`, and `Reaction Steps`, each holding
an object. These are text/JSON metrics, not the structure-level Table 2 metrics.

## Generate and train on a GPU

Use a separate Linux/CUDA environment with the reported training stack,
including the compatible Transformers development build, PyTorch, PEFT,
bitsandbytes, accelerate, and huggingface_hub. The saved-output comparison
above needs only Python. From this Qwen directory:

```bash
python scripts/download_model.py
python -m src.infer --data data/eval.jsonl --output tmp/baseline.jsonl --max-new-tokens 1024
python -m src.train_qlora --train-data data/train.jsonl --eval-data data/eval.jsonl --output outputs/qlora --epochs 5 --max-length 2048 --learning-rate 2e-4 --gradient-accumulation 4 --save-steps 14 --eval-steps 14 --seed 42
python -m src.infer --data data/eval.jsonl --adapter outputs/qlora/final --output tmp/finetuned.jsonl --max-new-tokens 1024
python -m src.compare --baseline tmp/baseline.jsonl --finetuned tmp/finetuned.jsonl --output tmp/comparison.md
```

The downloader pins both weights and processor files to the reported base-model
revision. `final` is saved after Trainer reloads the lowest-eval-loss checkpoint;
a new run may select a different step. For Slurm, adapt the resource request in
`scripts/qwen35_compare.sbatch`, activate the training environment, and submit
with `sbatch scripts/qwen35_compare.sbatch` from this directory. Set
`QWEN_RESUME=outputs/qlora/checkpoint-14` only if that complete checkpoint exists.

## Use the released checkpoint

Download [qwen3.5-9b-qlora-checkpoint-42.zip](https://github.com/snu-micc/uspto-revisit/releases/download/qwen3.5-9b-qlora-checkpoint-42/qwen3.5-9b-qlora-checkpoint-42.zip)
and verify its SHA-256 using [checkpoint-42/manifest.json](checkpoint-42/manifest.json).
Extract it into this Qwen directory; it contains a `checkpoint-42/` folder.
After downloading the pinned base model, run:

```bash
python -m src.infer --data data/eval.jsonl --adapter checkpoint-42 --output tmp/released-adapter.jsonl --max-new-tokens 1024
```

The release contains the original `adapter_model.safetensors` (43,278,336
parameters), tokenizer files, adapter configuration, and trainer-state JSON.
The tensors are stored as float32; BF16 in the settings above refers to training
compute precision. Only local path metadata was made portable, and the
author-reported base-model revision was added to the adapter configuration.
Weights are unchanged, with their original SHA-256 recorded in the manifest.
Optimizer, scheduler, RNG, and pickled training arguments are omitted from this
inference archive; it cannot replace a complete checkpoint for exact training
resume. The checkpoint's epoch-3 log verifies the best loss; the supplied
checkpoint-70 log verifies the final-epoch loss and the selection across all
five epochs.
