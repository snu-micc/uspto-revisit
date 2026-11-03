"""Run the 13 API configurations on newly drawn inputs, with durable resumption.

The six Qwen configurations run separately on the user's server. No provider
model is substituted if the recorded model is unavailable. Successful raw
predictions are preserved and bound to the input and prompt hashes.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import time
from pathlib import Path

CONFIGS = {
    **{f"gpt-5.4-{r}": ("openai", "gpt-5.4", r) for r in ("none", "low", "high")},
    **{f"gpt-5.6-sol-{r}": ("openai", "gpt-5.6-sol", r) for r in ("none", "low", "high")},
    "gpt-4.1-mini-nonreasoning-shared": ("openai", "gpt-4.1-mini", None),
    "gpt-4.1-mini-ft-nonreasoning-shared": ("openai", "ft:gpt-4.1-mini-2025-04-14:micc:uspto-revisit:DNfwY5a3", None),
    "gemini-2.5-flash-none": ("gemini", "gemini-2.5-flash", "none"),
    **{f"{m}-{r}": ("gemini", m, r) for m in ("gemini-3.1-pro-preview", "gemini-3.6-flash") for r in ("low", "high")},
}
MAX_OUTPUT_TOKENS = 16_384
FIELDS = ["idx", "benchmark_idx", "title", "paragraph", "prediction", "error", "provider", "model",
          "response_model", "reasoning_level", "input_tokens", "output_tokens", "reasoning_tokens",
          "latency_seconds", "attempt"]
ROOT = Path(__file__).resolve().parents[1]
SYSTEM = """You are a chemical reaction extraction assistant.
Return exactly one valid JSON object and nothing else.
Do not use markdown.
Do not use code fences.
Do not include any text before or after the JSON.

Follow the requested schema exactly.
Use only information supported by the input text.
Be careful to distinguish reaction steps from work-up steps."""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    temp = path.with_suffix(".partial")
    with temp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def load_env(path):
    if path:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def validate_json(raw):
    decoder = json.JSONDecoder()
    result, _ = decoder.raw_decode(raw.strip())
    return json.dumps(result, ensure_ascii=False)


async def run(args):
    load_env(args.dotenv)
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = read(args.input)
    if len({r["idx"] for r in inputs}) != len(inputs):
        raise ValueError("Repeated benchmark IDs")
    chosen = args.configurations or list(CONFIGS)
    settings = {k: CONFIGS[k] for k in chosen}
    identity = {"input_sha256": sha(args.input), "prompt_sha256": sha(args.prompt),
                "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "configurations": settings, "max_output_tokens": MAX_OUTPUT_TOKENS,
                "timeout_seconds": args.timeout, "temperature": "provider default"}
    manifest_path = args.output / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if json.dumps(previous["identity"], sort_keys=True) != json.dumps(identity, sort_keys=True):
            raise ValueError("Refusing to mix a different input, prompt or model configuration")
    manifest_path.write_text(json.dumps({"identity": identity, "status": "prepared"}, indent=2)+"\n", encoding="utf-8")
    if args.dry_run:
        print(f"Prepared {len(inputs)} rows x {len(chosen)} API configurations; no requests sent")
        return
    from openai import AsyncOpenAI
    from google import genai
    from google.genai import types
    clients = {}
    providers = {CONFIGS[k][0] for k in chosen}
    for provider in providers:
        key = os.environ.get("OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY")
        if not key:
            raise ValueError("Missing credential for " + provider)
        clients[provider] = AsyncOpenAI(api_key=key) if provider == "openai" else genai.Client(api_key=key).aio
    prompt = args.prompt.read_text(encoding="utf-8-sig")
    semaphore = asyncio.Semaphore(args.concurrency)
    async def one(config, index, record, attempt):
        provider, model, effort = CONFIGS[config]
        row = {k: "" for k in FIELDS}
        row.update(idx=index, benchmark_idx=record["idx"], title=record["title"], paragraph=record["paragraph"],
                   provider=provider, model=model, reasoning_level=effort or "", attempt=attempt)
        user = prompt.replace("{title}", record["title"]).replace("{paragraph}", record["paragraph"])
        start = time.monotonic()
        async with semaphore:
            try:
                if provider == "openai":
                    params = {"model": model, "instructions": SYSTEM, "input": user, "store": False,
                              "max_output_tokens": MAX_OUTPUT_TOKENS}
                    if effort is not None:
                        params.update(reasoning={"effort": effort}, text={"format": {"type": "json_object"}})
                    response = await asyncio.wait_for(clients[provider].responses.create(**params), args.timeout)
                    raw = response.output_text
                    row["response_model"] = response.model
                    usage = response.usage
                    if usage:
                        row.update(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                            reasoning_tokens=getattr(usage.output_tokens_details, "reasoning_tokens", ""))
                else:
                    thinking = types.ThinkingConfig(thinking_budget=0, include_thoughts=False) if effort == "none" else types.ThinkingConfig(thinking_level=effort, include_thoughts=False)
                    response = await asyncio.wait_for(clients[provider].models.generate_content(model=model, contents=user,
                        config=types.GenerateContentConfig(system_instruction=SYSTEM, response_mime_type="application/json",
                            thinking_config=thinking, max_output_tokens=MAX_OUTPUT_TOKENS)), args.timeout)
                    raw = response.text or ""
                    row["response_model"] = getattr(response, "model_version", "")
                    usage = response.usage_metadata
                    if usage:
                        row.update(input_tokens=usage.prompt_token_count, output_tokens=usage.candidates_token_count,
                                   reasoning_tokens=usage.thoughts_token_count)
                row["prediction"] = validate_json(raw)
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)
                for keyname in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
                    secret = os.environ.get(keyname)
                    if secret: error = error.replace(secret, "[REDACTED]")
                row["error"] = error
        row["latency_seconds"] = round(time.monotonic()-start, 3)
        return row
    # Each configuration is checkpointed as results arrive. A failed provider
    # does not erase completed requests for another model.
    reports = {}
    for config in chosen:
        path = args.output / (config + "_output.csv")
        records = {int(r["idx"]): r for r in read(path)} if path.exists() else {}
        todo = [i for i in range(len(inputs)) if i not in records or records[i]["error"] or not records[i]["prediction"]]
        tasks = [one(config, i, inputs[i], int(records.get(i, {}).get("attempt", 0))+1) for i in todo]
        for future in asyncio.as_completed(tasks):
            result = await future; records[int(result["idx"])] = result
            write(path, [records[i] for i in sorted(records)])
        errors = sum(bool(r["error"]) for r in records.values())
        reports[config] = {"rows": len(records), "errors": errors, "sha256": sha(path)}
        print(config, f"{len(records)-errors}/{len(inputs)} successful", flush=True)
    manifest_path.write_text(json.dumps({"identity": identity, "status": "complete" if all(not r["errors"] for r in reports.values()) else "incomplete",
        "reports": reports}, indent=2)+"\n", encoding="utf-8")
    for client in clients.values():
        try:
            await client.close() if hasattr(client, "close") else await client.aclose()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", type=Path, default=ROOT / "config/prompt.txt")
    p.add_argument("--dotenv", type=Path)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--configurations", nargs="+", choices=list(CONFIGS))
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
