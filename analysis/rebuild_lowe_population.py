"""Build a new, versioned flagged population directly from Lowe grant SMILES.

This is not a reconstruction of the undocumented historical CSV algorithm.
No historical Issue, Reagent, Template or TemplateFreq field is an input.
Run with --help; see LOWE_POPULATION.md for the exact rules and limitations.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing as mp
import math
import re
import subprocess
import sys
import time
import types
import urllib.request
from collections import Counter
from pathlib import Path

try:
    from analysis.benchmark_inputs import load_benchmark
except ModuleNotFoundError:
    from benchmark_inputs import load_benchmark

POLICY = "lowe-grants-raw-v3"
RAW_SHA256 = "5e91117efdb228642647ff26fc5f41b8babf544ca60b2967d76adf243352b626"
ARCHIVE_SHA256 = "d10f8ecb013d3f7bc85abdf6a8c5da8a7dd992931c1fda5db14fbd764343bc4e"
RAW_NAME = "1976_Sep2016_USPTOgrants_smiles.rsmi"
ARCHIVE_NAME = "1976_Sep2016_USPTOgrants_smiles.7z"
MAP = re.compile(r":(\d+)\]")
LABELS = {"many_products": "many_product (> 1)", "many_reactants": "many_reactant (> 2)",
          "many_reagents": "many_reagent (> 5)", "no_reagent": "no_reagent"}
SETTINGS = {"include_neighbors": False, "use_stereo": False, "use_symbol": True,
            "max_unmap": 5, "retro": False, "remote": False, "least_atom_num": 2,
            "verbose": False}
STAGE_FIELDS = ["RawRow", "Year", "PatentNumber", "ParagraphNum", "ReactionSmiles", "Reagent",
                "NReactants", "NReagents", "NProducts", "UnmappedProducts", "DuplicateKey",
                "Template", "NAtomChanges", "ProcessingStatus", "Error"]
FINAL_FIELDS = STAGE_FIELDS + ["TemplateFreq", "DuplicateOfRawRow", "NMolecules", "Issue", "Criteria", "FrameStratum", "Policy"]
_extract = None
MAX_TEMPLATE_PERMUTATIONS = 4096
csv.field_size_limit(2**31 - 1)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def roles(reaction):
    """Assign *whole* input molecules by positive product atom-map overlap.

    Unmapped product-side molecules are recorded, not reclassified as reagents.
    Empty components never contribute to a molecular count.
    """
    core = reaction.strip().split()[0]
    left, agents, right = core.split(">")
    products = [s for s in right.split(".") if s and MAP.search(s)]
    product_maps = set(MAP.findall(".".join(products)))
    inputs = [s for s in left.split(".") + agents.split(".") if s]
    reactants = [s for s in inputs if set(MAP.findall(s)) & product_maps]
    reagents = [s for s in inputs if not set(MAP.findall(s)) & product_maps]
    unmapped_products = [s for s in right.split(".") if s and not MAP.search(s)]
    return reactants, reagents, products, unmapped_products


def criteria(nr, ng, np):
    return [k for k, yes in [("many_products", np > 1), ("many_reactants", nr > 2),
                             ("many_reagents", ng > 5), ("no_reagent", ng == 0)] if yes]


def load_extractor():
    """Load the installed pure-RDKit extractor without importing DGL or a mapper.

    Namespace isolation does not modify the installed package or its source.
    """
    global _extract
    if _extract is None:
        dist = importlib.metadata.distribution("localmapper")
        folder = Path(dist.locate_file("localmapper/LocalTemplate"))
        package = types.ModuleType("_lowe_localtemplate")
        package.__path__ = [str(folder)]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + ".template_extractor")
        original = module.enumerate_mapping
        def bounded_mapping(transform):
            # The upstream enumerator materializes a Cartesian product. Bound
            # its size *before* allocating, independently of CPU speed/time.
            sizes = [len(module.permutations(component))
                     for side in transform.split(">>") for component in side.split(".")]
            if math.prod(sizes) > MAX_TEMPLATE_PERMUTATIONS:
                raise TemplateComplexityLimit("template atom-map permutations exceed 4096")
            return original(transform)
        module.enumerate_mapping = bounded_mapping
        _extract = module.extract_from_reaction
    return _extract


class TemplateComplexityLimit(RuntimeError):
    pass


def initialize():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    load_extractor()


def canonical_components(values):
    from rdkit import Chem
    result = []
    for value in values:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError("RDKit could not parse a molecular component")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        result.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return sorted(result)


def process_one(item):
    index, source = item
    row = {k: "" for k in STAGE_FIELDS}
    row.update({k: source.get(k, "") for k in ["Year", "PatentNumber", "ParagraphNum", "ReactionSmiles"]})
    row["RawRow"] = index
    try:
        reactants, reagents, products, unmapped = roles(row["ReactionSmiles"])
        row.update(NReactants=len(reactants), NReagents=len(reagents), NProducts=len(products),
                   UnmappedProducts=".".join(unmapped))
    except Exception as exc:
        row.update(ProcessingStatus="reaction_syntax_error", Error=str(exc))
        return row
    try:
        cr, cg, cp = map(canonical_components, [reactants, reagents, products])
        canonical_components(unmapped)  # validate, but do not turn RHS species into input reagents
        row["Reagent"] = ".".join(cg)
        if not cr or not cp:
            row.update(ProcessingStatus="missing_mapped_side")
            return row
        # Identity includes all three molecular roles, but not patent/paragraph.
        key = json.dumps([cr, cg, cp], separators=(",", ":"))
        row["DuplicateKey"] = hashlib.sha256(key.encode()).hexdigest()
    except Exception as exc:
        row.update(ProcessingStatus="molecular_parse_error", Error=str(exc))
        return row
    try:
        template = load_extractor()(".".join(reactants) + ">>" + ".".join(products), SETTINGS)
        row.update(Template=template or "", ProcessingStatus="ok" if template else "template_unassigned")
        row["NAtomChanges"] = len(set(MAP.findall(template))) if template else ""
    except TemplateComplexityLimit as exc:
        row.update(ProcessingStatus="template_complexity_limit", Error=str(exc))
    except Exception as exc:
        row.update(ProcessingStatus="template_error", Error=type(exc).__name__ + ": " + str(exc))
    return row


def process_batch(items):
    return [process_one(item) for item in items]


def source_batches(path, size, limit):
    with path.open(encoding="utf-8-sig", newline="") as f:
        batch = []
        for index, row in enumerate(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
            if limit and index >= limit:
                break
            batch.append((index, row))
            if len(batch) == size:
                yield index // size, batch
                batch = []
        if batch:
            yield batch[0][0] // size, batch


def prepare_source(args):
    if args.raw:
        return args.raw.resolve()
    rawdir = args.output / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    archive = args.archive
    if archive is None:
        archive = rawdir / ARCHIVE_NAME
        if not archive.exists():
            url = "https://api.figshare.com/v2/articles/5104873/versions/1"
            with urllib.request.urlopen(url) as f:
                article = json.load(f)
            dump(rawdir / "figshare_v1.json", article)
            entry = next(x for x in article["files"] if x["name"] == ARCHIVE_NAME)
            temp = archive.with_suffix(".download")
            urllib.request.urlretrieve(entry["download_url"], temp)
            temp.replace(archive)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA256 does not match the pinned figshare v1 source")
    raw = rawdir / RAW_NAME
    if not raw.exists():
        subprocess.run(["tar", "-xf", str(archive.resolve()), "-C", str(rawdir.resolve()), RAW_NAME], check=True)
    return raw


def stage_rows(folder):
    for path in sorted(folder.glob("*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)


def finalize(output, benchmark_path, manifest, membership_path=None):
    freq, first = Counter(), {}
    n = 0
    for row in stage_rows(output / "chunks"):
        n += 1
        if row["Template"]:
            freq[row["Template"]] += 1
        if row["DuplicateKey"]:
            first.setdefault(row["DuplicateKey"], int(row["RawRow"]))
    targets, matches = {}, {}
    if benchmark_path:
        benchmark = load_benchmark(benchmark_path, membership_path)
        for r in benchmark:
            targets.setdefault(r["Lowe_smiles"].strip().split()[0], []).append(r)
    counts, frame, status, labels = Counter(), Counter(), Counter(), Counter()
    flagged = 0
    fields = FINAL_FIELDS
    with (output / "grants_issue.csv").open("w", encoding="utf-8", newline="") as f, gzip.open(output / "sampling_frame.csv.gz", "wt", encoding="utf-8", newline="") as sf:
        writer, fw = csv.DictWriter(f, fieldnames=fields), csv.DictWriter(sf, fieldnames=fields)
        writer.writeheader(); fw.writeheader()
        for row in stage_rows(output / "chunks"):
            met = criteria(int(row["NReactants"]), int(row["NReagents"]), int(row["NProducts"])) if row["NReactants"] != "" else []
            counts.update(met); counts["union"] += bool(met)
            tf = freq[row["Template"]] if row["Template"] else ""
            duplicate_of = first[row["DuplicateKey"]] if row["DuplicateKey"] else None
            duplicate = duplicate_of is not None and duplicate_of != int(row["RawRow"])
            issue = [LABELS[k] for k in met]
            if tf == 1:
                issue.append("rare_temp")
            if duplicate:
                issue.append("duplicated")
            if row["ProcessingStatus"] != "ok":
                issue.append(row["ProcessingStatus"])
            # Global duplicates and template-processing failures are diagnostic
            # flags, not reasons to replace benchmark records. Keep the existing
            # singleton-template exclusion; an unassigned template is not rare.
            stratum = met[0] if len(met) == 1 and tf != 1 else ""
            row.update(TemplateFreq=tf, DuplicateOfRawRow=duplicate_of if duplicate else "",
                       NMolecules=str(tuple(int(row[k]) if row[k] != "" else "" for k in ("NReactants", "NReagents", "NProducts"))),
                       Issue=repr(issue), Criteria="+".join(met), FrameStratum=stratum, Policy=POLICY)
            if issue:
                writer.writerow(row); flagged += 1
            if stratum:
                frame[stratum] += 1; fw.writerow(row)
            status[row["ProcessingStatus"]] += 1; labels.update(issue)
            for r in targets.get(row["ReactionSmiles"].strip().split()[0], []):
                idx = r.get("idx", str(benchmark.index(r)))
                patent = r.get("patent_number", "")
                if patent and row["PatentNumber"] != patent:
                    continue
                m = matches.setdefault(idx, {"idx": idx, "assigned_stratum": r.get("stratum", ""),
                    "raw_rows": [], "criteria": row["Criteria"], "n_reactants": row["NReactants"],
                    "n_reagents": row["NReagents"], "n_products": row["NProducts"],
                    "template_frequency": tf, "eligible_raw_rows": [], "reasons": set()})
                m["raw_rows"].append(int(row["RawRow"]))
                if stratum == m["assigned_stratum"]:
                    m["eligible_raw_rows"].append(int(row["RawRow"]))
                if met != [m["assigned_stratum"]]: m["reasons"].add("criteria_mismatch")
                if tf == 1: m["reasons"].add("rare_temp")
    audit = []
    if benchmark_path:
        for i, r in enumerate(benchmark):
            idx = r.get("idx", str(i))
            m = matches.get(idx, {"idx": idx, "assigned_stratum": r.get("stratum", ""), "raw_rows": [],
                                 "eligible_raw_rows": [], "reasons": {"not_found_in_source"}})
            m["eligible"] = bool(m["eligible_raw_rows"])
            m["reasons"] = "+".join(sorted(m["reasons"]))
            m["raw_rows"] = ";".join(map(str, m["raw_rows"]))
            m["eligible_raw_rows"] = ";".join(map(str, m["eligible_raw_rows"]))
            audit.append(m)
        af = list(dict.fromkeys(k for a in audit for k in a))
        with (output / "benchmark_membership.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=af); w.writeheader(); w.writerows(audit)
    summary = {"policy": POLICY, "complete_source": n == 1808937 and not manifest["limit"],
               "source_rows": n, "flagged_rows": flagged, "table1": dict(counts),
               "table1_percent": {k: v / n * 100 for k, v in counts.items()},
               "sampling_frame": dict(frame), "sampling_frame_total": sum(frame.values()),
               "processing_status": dict(status), "issue_counts": dict(labels),
               "benchmark_rows": len(audit), "benchmark_eligible": sum(x["eligible"] for x in audit),
               "benchmark_ineligible_ids": [x["idx"] for x in audit if not x["eligible"]]}
    dump(output / "summary.json", summary)
    with (output / "table1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["criterion", "n", "denominator", "percent", "policy"])
        for k in list(LABELS) + ["union"]:
            w.writerow([k, counts[k], n, f"{100 * counts[k] / n:.8f}", POLICY])
    manifest.update(status="complete" if summary["complete_source"] else "partial", output_sha256={
        p.name: sha256(p) for p in [output / "grants_issue.csv", output / "sampling_frame.csv.gz", output / "table1.csv", output / "summary.json"]})
    dump(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--benchmark-membership", type=Path,
                        help="Optional metadata for older split inputs; omit for the unified input.csv.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, help="Pilot only: never accepted as a complete corpus")
    parser.add_argument("--allow-unpinned-source", action="store_true", help="For synthetic tests only")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = prepare_source(args)
    checksum = sha256(raw)
    if checksum != RAW_SHA256 and not args.allow_unpinned_source:
        raise ValueError("Raw source SHA256 does not match Lowe figshare v1")
    from rdkit import rdBase
    dist = importlib.metadata.distribution("localmapper")
    folder = Path(dist.locate_file("localmapper/LocalTemplate"))
    source_hashes = {p.name: sha256(p) for p in sorted(folder.glob("*.py"))}
    manifest = {"policy": POLICY, "raw_sha256": checksum, "raw_path": str(raw.resolve()),
                "builder_sha256": sha256(Path(__file__)), "rdkit": rdBase.rdkitVersion,
                "localmapper": dist.version, "extractor_sha256": source_hashes, "settings": SETTINGS,
                "max_template_permutations": MAX_TEMPLATE_PERMUTATIONS,
                "batch_size": args.batch_size, "limit": args.limit, "status": "running"}
    old = args.output / "manifest.json"
    if old.exists():
        previous = json.loads(old.read_text(encoding="utf-8"))
        for key in ["policy", "raw_sha256", "builder_sha256", "rdkit", "localmapper", "extractor_sha256", "settings", "max_template_permutations", "batch_size", "limit"]:
            if previous.get(key) != manifest[key]:
                raise ValueError("Resume manifest differs: " + key + "; use a fresh output directory")
    dump(old, manifest)
    chunks = args.output / "chunks"; chunks.mkdir(exist_ok=True)
    started = time.monotonic()
    def pending():
        for number, batch in source_batches(raw, args.batch_size, args.limit):
            if not (chunks / f"{number:07d}.csv.gz").exists():
                yield number, batch
    # imap is lazy and bounds the queued work, unlike materializing 1.8M jobs.
    with mp.get_context("spawn").Pool(args.workers, initializer=initialize) as pool:
        numbers = []
        def batches():
            for number, batch in pending():
                numbers.append(number)
                yield batch
        for j, rows in enumerate(pool.imap(process_batch, batches(), chunksize=1)):
            number = numbers[j]
            path = chunks / f"{number:07d}.csv.gz"
            temp = path.with_suffix(".partial")
            with gzip.open(temp, "wt", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=STAGE_FIELDS); w.writeheader(); w.writerows(rows)
            temp.replace(path)
            if j % 20 == 0:
                print(f"Completed chunk {number}; new rows {(j+1)*args.batch_size:,}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    finalize(args.output, args.benchmark, manifest, args.benchmark_membership)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
