"""CLI for the USPTO Revisit reaction-SMILES pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import ssl
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from uspto_revisit.atom_mapping import add_atom_mapping_columns
from uspto_revisit.file_io import ensure_directory, save_smiles_dict
from uspto_revisit.gpt_extract import (
    DEFAULT_SYSTEM_PROMPT,
    load_prompt,
    results_to_frame,
    run_all_prompt_json,
)
from uspto_revisit.nosmi_review import (
    apply_review,
    build_review_queue,
    model_specs_from_values,
)
from uspto_revisit.reaction_smiles import process_smiles_data
from uspto_revisit.smiles_fetch import (
    audit_name_smiles_consistency,
    load_cache,
    load_resolution_overrides,
    process_batch,
    reprocess_no_smi,
    save_cache,
)


def default_no_smi_overrides_path() -> str | None:
    """Return the repository override file when it is available."""
    candidates = (
        Path("config/nosmi_overrides.json"),
        Path(__file__).resolve().parents[2] / "config" / "nosmi_overrides.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


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


def aiohttp_ssl_context() -> ssl.SSLContext:
    """Use certifi CA roots so asynchronous lookups work on Windows."""
    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def build_parser() -> argparse.ArgumentParser:
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Resolve GPT reaction JSON outputs to reaction SMILES.",
    )
    subparsers = parser.add_subparsers(dest="command")

    gpt_parser = subparsers.add_parser(
        "gpt-extract",
        help="Generate structured reaction JSON with OpenAI or Gemini.",
    )
    gpt_parser.add_argument(
        "--input",
        default="input.csv",
        help="Input CSV or XLSX path. Default: input.csv",
    )
    gpt_parser.add_argument(
        "--output",
        default=None,
        help="Output CSV or XLSX path. Default: result/{selected model}_output.csv",
    )
    gpt_parser.add_argument(
        "--provider",
        choices=("openai", "gemini"),
        default=os.getenv("LLM_PROVIDER", "openai").lower(),
        help="LLM provider. Default: LLM_PROVIDER or openai",
    )
    gpt_parser.add_argument(
        "--model",
        default=None,
        help="Model name. Default: OPENAI_MODEL or GEMINI_MODEL for the provider",
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
        help="Concurrent LLM requests. Default: 10",
    )
    gpt_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20,
        help="Timeout per LLM request. Default: 20",
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

    atom_map_parser = subparsers.add_parser(
        "atom-map",
        help="Atom-map an existing reaction-SMILES CSV or XLSX with LocalMapper.",
    )
    atom_map_parser.add_argument(
        "--input",
        required=True,
        help="Reaction-SMILES CSV or XLSX path.",
    )
    atom_map_parser.add_argument(
        "--reaction-column",
        default=None,
        help="Reaction SMILES column. Inferred when exactly one *_rxn column exists.",
    )
    atom_map_parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for LocalMapper result columns. Default: reaction column prefix.",
    )
    atom_map_parser.add_argument(
        "--output",
        default=None,
        help="Mapped output path. Default: <input_stem>_mapped.<extension>",
    )
    atom_map_parser.add_argument(
        "--device",
        default="cpu",
        help="LocalMapper PyTorch device. Default: cpu",
    )
    atom_map_parser.add_argument(
        "--model-version",
        default="202403",
        help="Bundled LocalMapper model version. Default: 202403",
    )
    atom_map_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of reactions per LocalMapper inference batch. Default: 32",
    )

    retry_parser = subparsers.add_parser(
        "retry-nosmi",
        help=(
            "Retry only unresolved (NoSmi) names in an existing reaction-SMILES "
            "result and optionally rerun LocalMapper."
        ),
    )
    retry_parser.add_argument(
        "--input",
        required=True,
        help="Existing reaction-SMILES CSV or XLSX containing model predictions.",
    )
    retry_parser.add_argument(
        "--smiles-dict",
        required=True,
        help="Existing smiles_dict_final.json whose NoSmi entries will be retried.",
    )
    retry_parser.add_argument(
        "--output-smiles-dict",
        default=None,
        help="Recovered dictionary path. Default: <smiles-dict-stem>_recovered.json",
    )
    retry_parser.add_argument(
        "--model-column",
        default="prediction",
        help="Column containing structured reaction JSON. Default: prediction",
    )
    retry_parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix of the smiles, skeleton, reaction, and mapping columns.",
    )
    retry_parser.add_argument(
        "--output",
        default=None,
        help="Recovered result path. Default: <input-stem>_recovered.<extension>",
    )
    retry_parser.add_argument(
        "--cache",
        default=None,
        help="Optional shared pickle cache for successful recovery lookups.",
    )
    retry_parser.add_argument(
        "--overrides",
        default=default_no_smi_overrides_path(),
        help=(
            "Optional JSON file with auditable dataset-specific lookup aliases or "
            "active-component SMILES. Default: config/nosmi_overrides.json when present."
        ),
    )
    retry_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Rows saved per recovery batch. Default: 100",
    )
    retry_parser.add_argument(
        "--reprocess-concurrency",
        type=int,
        default=6,
        help="Concurrent external name lookups. Default: 6",
    )
    retry_parser.add_argument(
        "--log-file",
        default="result/nosmi_recovery.log",
        help="Recovery log file. Default: result/nosmi_recovery.log",
    )
    retry_parser.add_argument(
        "--map-atoms",
        action="store_true",
        help="Rerun LocalMapper after rebuilding reaction SMILES.",
    )
    retry_parser.add_argument(
        "--mapping-device",
        default="cpu",
        help="LocalMapper PyTorch device used with --map-atoms. Default: cpu",
    )
    retry_parser.add_argument(
        "--mapping-model-version",
        default="202403",
        help="Bundled LocalMapper model version. Default: 202403",
    )
    retry_parser.add_argument(
        "--mapping-batch-size",
        type=int,
        default=32,
        help="LocalMapper inference batch size. Default: 32",
    )

    review_export_parser = subparsers.add_parser(
        "nosmi-review-export",
        help="Export unresolved NoSmi compounds for row-scoped human review.",
    )
    review_export_parser.add_argument(
        "--result-dir",
        default="result",
        help="Directory used to discover complete model outputs. Default: result",
    )
    review_export_parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("PREFIX", "CSV", "SMILES_DICT"),
        help=(
            "Optional explicit model specification. Repeat for each model. "
            "When omitted, complete final outputs are discovered under --result-dir."
        ),
    )
    review_export_parser.add_argument(
        "--audit",
        default="result/nosmi_recovery_audit.csv",
        help="Automatic NoSmi audit used for status and candidate information.",
    )
    review_export_parser.add_argument(
        "--output",
        default="result/nosmi_human_review.csv",
        help="Editable human-review table. Default: result/nosmi_human_review.csv",
    )

    review_apply_parser = subparsers.add_parser(
        "nosmi-review-apply",
        help="Validate and apply row-scoped human NoSmi decisions.",
    )
    review_apply_parser.add_argument(
        "--result-dir",
        default="result",
        help="Directory used to discover complete model outputs. Default: result",
    )
    review_apply_parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("PREFIX", "CSV", "SMILES_DICT"),
        help=(
            "Optional explicit model specification. Repeat for each model. "
            "When omitted, complete final outputs are discovered under --result-dir."
        ),
    )
    review_apply_parser.add_argument(
        "--review",
        default="result/nosmi_human_review.csv",
        help="Completed human-review CSV or XLSX file.",
    )
    review_apply_parser.add_argument(
        "--output-dir",
        default="result",
        help="Directory for rebuilt model outputs. Default: result",
    )
    review_apply_parser.add_argument(
        "--summary",
        default="result/nosmi_human_review_summary.json",
        help="Application summary JSON.",
    )
    review_apply_parser.add_argument(
        "--map-atoms",
        action="store_true",
        help="Rerun LocalMapper after applying reviewed SMILES.",
    )
    review_apply_parser.add_argument(
        "--mapping-device",
        default="cpu",
        help="LocalMapper PyTorch device used with --map-atoms. Default: cpu",
    )
    review_apply_parser.add_argument(
        "--mapping-model-version",
        default="202403",
        help="Bundled LocalMapper model version. Default: 202403",
    )
    review_apply_parser.add_argument(
        "--mapping-batch-size",
        type=int,
        default=32,
        help="LocalMapper inference batch size. Default: 32",
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
        help="Final output CSV path. Default: result/{prefix}_reaction_smiles_final.csv",
    )
    parser.add_argument(
        "--with-smiles-output",
        default=None,
        help="Optional intermediate CSV with SMILES dictionaries. Not saved unless provided.",
    )
    parser.add_argument(
        "--batch-dir",
        default=None,
        help=(
            "Directory for SMILES dictionaries and temporary lookup files. "
            "Default: result/smiles_batches/{prefix}"
        ),
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
        help="Skip PubChem/OPSIN/CIR/ChEBI/ChemSpider retry for unresolved compounds.",
    )
    parser.add_argument(
        "--overrides",
        default=default_no_smi_overrides_path(),
        help=(
            "JSON file with auditable name aliases or active-component SMILES used "
            "during final NoSmi recovery. Default: config/nosmi_overrides.json when present."
        ),
    )
    parser.add_argument(
        "--fix-names",
        action="store_true",
        help="Normalize compound names before the first lookup pass.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path. Default: result/{prefix}_smiles_fetch.log",
    )
    parser.add_argument(
        "--map-atoms",
        action="store_true",
        help="Run LocalMapper after reaction SMILES generation.",
    )
    parser.add_argument(
        "--mapping-device",
        default="cpu",
        help="LocalMapper PyTorch device used with --map-atoms. Default: cpu",
    )
    parser.add_argument(
        "--mapping-model-version",
        default="202403",
        help="Bundled LocalMapper model version. Default: 202403",
    )
    parser.add_argument(
        "--mapping-batch-size",
        type=int,
        default=32,
        help="LocalMapper inference batch size. Default: 32",
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


def default_model_for_provider(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def default_reaction_smiles_output_path(output_prefix: str) -> Path:
    return Path("result") / (
        f"{safe_model_filename(output_prefix)}_reaction_smiles_final.csv"
    )


def default_atom_mapping_output_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    suffix = path.suffix or ".csv"
    return path.with_name(f"{path.stem}_mapped{suffix}")


def default_recovery_output_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    suffix = path.suffix or ".csv"
    return path.with_name(f"{path.stem}_recovered{suffix}")


def default_recovered_dict_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_recovered{suffix}")


def infer_reaction_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in frame.columns:
            available = ", ".join(frame.columns)
            raise ValueError(
                f"Column '{requested}' was not found. Available columns: {available}"
            )
        return requested

    candidates = [
        column
        for column in frame.columns
        if column.endswith("_rxn")
        and not column.endswith(("_mapped_rxn", "_localmapper_rxn"))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "No reaction column was found. Pass --reaction-column explicitly."
        )
    raise ValueError(
        "Multiple reaction columns were found: "
        f"{', '.join(candidates)}. Pass --reaction-column explicitly."
    )


def run_atom_mapping(args: argparse.Namespace) -> Path:
    frame = read_table(args.input)
    reaction_column = infer_reaction_column(frame, args.reaction_column)
    output_prefix = args.output_prefix or reaction_column.removesuffix("_rxn")
    output_path = (
        Path(args.output)
        if args.output
        else default_atom_mapping_output_path(args.input)
    )
    mapped_frame = add_atom_mapping_columns(
        frame,
        reaction_column,
        output_prefix,
        device=args.device,
        model_version=args.model_version,
        batch_size=args.batch_size,
    )
    write_table(mapped_frame, output_path)
    return output_path


async def run_retry_no_smi(args: argparse.Namespace) -> Path:
    """Retry existing NoSmi entries without repeating GPT or all-name lookups."""
    import aiohttp

    input_path = Path(args.input)
    smiles_dict_path = Path(args.smiles_dict)
    recovered_dict_path = (
        Path(args.output_smiles_dict)
        if args.output_smiles_dict
        else default_recovered_dict_path(smiles_dict_path)
    )
    output_path = (
        Path(args.output)
        if args.output
        else default_recovery_output_path(input_path)
    )

    ensure_directory(recovered_dict_path.parent)
    ensure_directory(output_path.parent)
    ensure_directory(Path(args.log_file).parent)
    logging.basicConfig(
        filename=args.log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    frame = read_table(input_path)
    if args.model_column not in frame.columns:
        available = ", ".join(frame.columns)
        raise ValueError(
            f"Column '{args.model_column}' was not found in {input_path}. "
            f"Available columns: {available}"
        )

    if args.cache:
        cache_path = Path(args.cache)
        ensure_directory(cache_path.parent)
        load_cache(cache_path)
    else:
        cache_path = None

    load_resolution_overrides(args.overrides)

    semaphore = asyncio.Semaphore(args.reprocess_concurrency)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=aiohttp_ssl_context())
    ) as session:
        await reprocess_no_smi(
            smiles_dict_path,
            recovered_dict_path,
            session,
            semaphore,
            batch_size=args.batch_size,
        )
    if cache_path:
        save_cache(cache_path)

    with recovered_dict_path.open("r", encoding="utf-8-sig") as handle:
        smiles_dicts = json.load(handle)
    if len(smiles_dicts) != len(frame):
        raise ValueError(
            f"Recovered SMILES dictionaries ({len(smiles_dicts)}) do not match "
            f"input rows ({len(frame)})."
        )

    prefix = args.output_prefix
    smiles_column = f"{prefix}_smiles"
    reaction_column = f"{prefix}_rxn"
    frame[smiles_column] = smiles_dicts
    skeleton_smiles, final_smiles, _errors = process_smiles_data(
        frame[args.model_column].fillna("").astype(str).tolist(),
        smiles_dicts,
    )
    frame[f"{prefix}_skeleton"] = skeleton_smiles
    frame[reaction_column] = final_smiles
    if args.map_atoms:
        frame = add_atom_mapping_columns(
            frame,
            reaction_column,
            prefix,
            device=args.mapping_device,
            model_version=args.mapping_model_version,
            batch_size=args.mapping_batch_size,
        )
    write_table(frame, output_path)
    logging.info("NoSmi recovery completed. Output written to %s", output_path)
    return output_path


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
    overrides: str | Path | None = None,
) -> Path:
    import aiohttp

    initial_path = batch_dir / "smiles_dict_initial.json"
    final_path = batch_dir / "smiles_dict_final.json"
    if skip_reprocess:
        return initial_path

    load_resolution_overrides(overrides)
    semaphore = asyncio.Semaphore(reprocess_concurrency)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=aiohttp_ssl_context())
    ) as session:
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
    safe_prefix = safe_model_filename(output_prefix)
    batch_dir = (
        Path(args.batch_dir)
        if args.batch_dir
        else Path("result") / "smiles_batches" / safe_prefix
    )
    log_path = (
        Path(args.log_file)
        if args.log_file
        else Path("result") / f"{safe_prefix}_smiles_fetch.log"
    )

    ensure_directory(output_path.parent)
    if with_smiles_output_path:
        ensure_directory(with_smiles_output_path.parent)
    ensure_directory(batch_dir)
    ensure_directory(log_path.parent)

    logging.basicConfig(
        filename=log_path,
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
        overrides=args.overrides,
    )
    conflict_count = audit_name_smiles_consistency(
        batch_dir / "smiles_cache.pkl",
        batch_dir / "smiles_consistency_audit.csv",
    )
    if conflict_count:
        logging.warning("Found %s same-name SMILES conflicts; see smiles_consistency_audit.csv", conflict_count)

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
    if args.map_atoms:
        output_frame = add_atom_mapping_columns(
            output_frame,
            f"{output_prefix}_rxn",
            output_prefix,
            device=args.mapping_device,
            model_version=args.mapping_model_version,
            batch_size=args.mapping_batch_size,
        )
    output_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info("Pipeline completed. Output written to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "nosmi-review-export":
        try:
            specs = model_specs_from_values(args.model, args.result_dir)
            output_path = Path(args.output)
            review = build_review_queue(
                specs,
                audit_path=args.audit,
                existing_review_path=output_path,
            )
            write_table(review, output_path)
        except Exception as exc:
            parser.exit(1, f"Error: {exc}\n")
        print(
            f"NoSmi human-review table created: {output_path} "
            f"({len(review)} row-scoped compounds)"
        )
        return 0

    if args.command == "nosmi-review-apply":
        try:
            specs = model_specs_from_values(args.model, args.result_dir)
            summary = apply_review(
                specs,
                review_path=args.review,
                output_dir=args.output_dir,
                summary_path=args.summary,
                map_atoms=args.map_atoms,
                mapping_device=args.mapping_device,
                mapping_model_version=args.mapping_model_version,
                mapping_batch_size=args.mapping_batch_size,
            )
        except Exception as exc:
            parser.exit(1, f"Error: {exc}\n")
        print(
            "Human NoSmi review applied: "
            f"{summary['matched_reviewed_row_names']} row-scoped compounds; "
            f"summary={args.summary}"
        )
        return 0

    if args.command == "atom-map":
        try:
            output_path = run_atom_mapping(args)
        except Exception as exc:
            parser.exit(1, f"Error: {exc}\n")
        print(f"Atom mapping completed: {output_path}")
        return 0

    if args.command == "retry-nosmi":
        try:
            output_path = asyncio.run(run_retry_no_smi(args))
        except Exception as exc:
            parser.exit(1, f"Error: {exc}\n")
        print(f"NoSmi recovery completed: {output_path}")
        return 0

    if args.command == "gpt-extract":
        try:
            model_name = args.model or default_model_for_provider(args.provider)
            output_path = (
                Path(args.output)
                if args.output
                else default_gpt_output_path(model_name)
            )
            ensure_directory(output_path.parent)
            input_df = read_table(args.input)
            if args.limit is not None:
                input_df = input_df.head(args.limit)
            prompt = load_prompt(args.prompt)
            results = asyncio.run(
                run_all_prompt_json(
                    input_df=input_df,
                    model_name=model_name,
                    provider=args.provider,
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
        print(f"LLM extraction completed: {output_path}")
        return 0

    try:
        output_path = asyncio.run(run_pipeline(args))
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Pipeline completed: {output_path}")
    return 0
