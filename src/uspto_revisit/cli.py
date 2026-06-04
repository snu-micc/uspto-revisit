"""CLI for the USPTO Revisit reaction-SMILES pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm

from uspto_revisit.file_io import ensure_directory, save_smiles_dict
from uspto_revisit.gpt_extract import (
    DEFAULT_SYSTEM_PROMPT,
    load_prompt,
    results_to_frame,
    run_all_prompt_json,
)
from uspto_revisit.reaction_smiles import process_smiles_data
from uspto_revisit.smiles_fetch import (
    load_cache,
    process_batch,
    reprocess_no_smi,
    save_cache,
)


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env without overriding existing env vars."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not value.startswith(("'", '"')) and "#" in value:
            value = value.split("#", 1)[0].strip()
        value = value.strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Resolve GPT reaction JSON outputs to reaction SMILES.",
    )
    subparsers = parser.add_subparsers(dest="command")

    gpt_parser = subparsers.add_parser(
        "gpt-extract",
        help="Generate structured reaction JSON from title/paragraph columns.",
    )
    gpt_parser.add_argument(
        "--input",
        default="input.csv",
        help="Input CSV or XLSX path. Default: input.csv",
    )
    gpt_parser.add_argument(
        "--output",
        default=None,
        help="Output CSV or XLSX path. Default: result/{OPENAI_MODEL}_output.csv",
    )
    gpt_parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model name. Default: OPENAI_MODEL or gpt-4.1-mini",
    )
    gpt_parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt template path. Default: prompts/prompt.txt",
    )
    gpt_parser.add_argument(
        "--title-column",
        default="title",
        help="Input title column. Default: title",
    )
    gpt_parser.add_argument(
        "--paragraph-column",
        default="paragraph",
        help="Input paragraph column. Default: paragraph",
    )
    gpt_parser.add_argument(
        "--semaphore-size",
        type=int,
        default=10,
        help="Concurrent OpenAI requests. Default: 10",
    )
    gpt_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20,
        help="Timeout per OpenAI request. Default: 20",
    )
    gpt_parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt text.",
    )
    gpt_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N rows.",
    )

    parser.add_argument(
        "--input",
        default="data/GPT_response.csv",
        help="Input CSV path. Default: data/GPT_response.csv",
    )
    parser.add_argument(
        "--model-column",
        default="GPT_finetuned_five",
        help="CSV column containing structured reaction JSON. Default: GPT_finetuned_five",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Prefix for generated result columns. "
            "Default: OPENAI_MODEL if set, otherwise --model-column."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Final output CSV path. Default: result/{OPENAI_MODEL}_reaction_smiles.csv",
    )
    parser.add_argument(
        "--with-smiles-output",
        default=None,
        help="Optional intermediate CSV with SMILES dictionaries. Not saved unless provided.",
    )
    parser.add_argument(
        "--batch-dir",
        default="result/smiles_batches",
        help="Directory for intermediate SMILES dictionary files. Default: result/smiles_batches",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows per lookup batch. Default: 100",
    )
    parser.add_argument(
        "--lookup-concurrency",
        type=int,
        default=60,
        help="Concurrent requests for initial OPSIN lookup. Default: 60",
    )
    parser.add_argument(
        "--reprocess-concurrency",
        type=int,
        default=40,
        help="Concurrent requests for NoSmi reprocessing. Default: 40",
    )
    parser.add_argument(
        "--skip-reprocess",
        action="store_true",
        help="Skip PubChem/CIR/ChemSpider retry pass for unresolved compounds.",
    )
    parser.add_argument(
        "--fix-names",
        action="store_true",
        help="Normalize compound names before the first lookup pass.",
    )
    parser.add_argument(
        "--log-file",
        default="result/smiles_fetch.log",
        help="Log file path. Default: result/smiles_fetch.log",
    )
    return parser


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(table_path)
    return pd.read_csv(table_path)


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    table_path = Path(path)
    ensure_directory(table_path.parent)
    if table_path.suffix.lower() in {".xlsx", ".xls"}:
        frame.to_excel(table_path, index=False)
    else:
        frame.to_csv(table_path, index=False, encoding="utf-8-sig")


def safe_model_filename(model_name: str) -> str:
    """Convert model names, including fine-tuned IDs, into safe filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip())
    return cleaned.strip("._-") or "gpt_output"


def default_gpt_output_path(model_name: str) -> Path:
    return Path("result") / f"{safe_model_filename(model_name)}_output.csv"


def default_reaction_smiles_output_path(output_prefix: str) -> Path:
    return Path("result") / f"{safe_model_filename(output_prefix)}_reaction_smiles.csv"


async def make_smiles_dict(
    responses,
    batch_dir: Path,
    batch_size: int,
    fix_names: bool,
    lookup_concurrency: int,
):
    ensure_directory(batch_dir)
    cache_path = batch_dir / "smiles_cache.pkl"
    load_cache(cache_path)

    smiles_dicts = []
    semaphore = asyncio.Semaphore(lookup_concurrency)
    total_batches = (len(responses) + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, len(responses), batch_size),
        desc="Resolving SMILES",
        unit="batch",
    ):
        batch = responses[start : start + batch_size]
        batch_number = start // batch_size + 1
        try:
            batch_smiles = await process_batch(batch, fix_name_bool=fix_names, semaphore=semaphore)
            smiles_dicts.extend(batch_smiles)
            logging.info("Completed batch %s/%s", batch_number, total_batches)
            save_smiles_dict(smiles_dicts, batch_dir / f"smiles_dict_batch_{batch_number}.json")
            save_cache(cache_path)
        except Exception as exc:
            logging.error("Error in batch %s: %s", batch_number, exc)
            smiles_dicts.extend(["Error"] * len(batch))

    save_smiles_dict(smiles_dicts, batch_dir / "smiles_dict_initial.json")
    return smiles_dicts


async def maybe_reprocess_no_smi(
    batch_dir: Path,
    batch_size: int,
    reprocess_concurrency: int,
    skip_reprocess: bool,
) -> Path:
    initial_path = batch_dir / "smiles_dict_initial.json"
    final_path = batch_dir / "smiles_dict_final.json"
    if skip_reprocess:
        return initial_path

    semaphore = asyncio.Semaphore(reprocess_concurrency)
    async with aiohttp.ClientSession() as session:
        await reprocess_no_smi(
            initial_path,
            final_path,
            session,
            semaphore,
            batch_size=batch_size,
        )
    return final_path


async def run_pipeline(args: argparse.Namespace) -> Path:
    input_path = Path(args.input)
    output_prefix = args.output_prefix or os.getenv("OPENAI_MODEL") or args.model_column
    output_path = (
        Path(args.output)
        if args.output
        else default_reaction_smiles_output_path(output_prefix)
    )
    with_smiles_output_path = (
        Path(args.with_smiles_output)
        if args.with_smiles_output
        else None
    )
    batch_dir = Path(args.batch_dir)

    ensure_directory(output_path.parent)
    if with_smiles_output_path:
        ensure_directory(with_smiles_output_path.parent)
    ensure_directory(batch_dir)
    ensure_directory(Path(args.log_file).parent)

    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    frame = pd.read_csv(input_path)
    if args.model_column not in frame.columns:
        available = ", ".join(frame.columns)
        raise ValueError(
            f"Column '{args.model_column}' was not found in {input_path}. "
            f"Available columns: {available}"
        )

    responses = frame[args.model_column].fillna("").astype(str).tolist()
    await make_smiles_dict(
        responses,
        batch_dir=batch_dir,
        batch_size=args.batch_size,
        fix_names=args.fix_names,
        lookup_concurrency=args.lookup_concurrency,
    )
    smiles_dict_path = await maybe_reprocess_no_smi(
        batch_dir=batch_dir,
        batch_size=args.batch_size,
        reprocess_concurrency=args.reprocess_concurrency,
        skip_reprocess=args.skip_reprocess,
    )

    with smiles_dict_path.open("r", encoding="utf-8-sig") as handle:
        smiles_dicts = json.load(handle)
    smiles_column = f"{output_prefix}_smiles"
    frame[smiles_column] = smiles_dicts
    if with_smiles_output_path:
        frame.to_csv(with_smiles_output_path, index=False, encoding="utf-8-sig")

    skeleton_smiles, final_smiles, _errors = process_smiles_data(
        frame[args.model_column].tolist(),
        frame[smiles_column].tolist(),
    )
    frame[f"{output_prefix}_skeleton"] = skeleton_smiles
    frame[f"{output_prefix}_rxn"] = final_smiles
    output_frame = frame.drop(
        columns=[column for column in ("idx", "model") if column in frame.columns]
    )
    output_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info("Pipeline completed. Output written to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gpt-extract":
        try:
            output_path = Path(args.output) if args.output else default_gpt_output_path(args.model)
            ensure_directory(output_path.parent)
            input_df = read_table(args.input)
            if args.limit is not None:
                input_df = input_df.head(args.limit)
            prompt = load_prompt(args.prompt)
            results = asyncio.run(
                run_all_prompt_json(
                    input_df=input_df,
                    model_name=args.model,
                    gpt_prompt=prompt,
                    system_prompt=args.system_prompt,
                    title_column=args.title_column,
                    paragraph_column=args.paragraph_column,
                    semaphore_size=args.semaphore_size,
                    timeout_seconds=args.timeout_seconds,
                    partial_output_path=output_path,
                )
            )
            output_frame = results_to_frame(results)
            write_table(output_frame, output_path)
        except Exception as exc:
            parser.exit(1, f"Error: {exc}\n")
        print(f"GPT extraction completed: {output_path}")
        return 0

    try:
        output_path = asyncio.run(run_pipeline(args))
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Pipeline completed: {output_path}")
    return 0
