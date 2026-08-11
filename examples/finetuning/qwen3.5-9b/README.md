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

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
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
