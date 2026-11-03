"""Draw source-linked replacement inputs; never invent or approve ground truth.

The existing benchmark is preserved outside the output directory. Replacement
rows are drawn uniformly within each new stratum, before inspecting outcomes.
This partial redraw does not make the retained historical sample a probability
sample of the expanded new frame. Population extrapolation is gated separately.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from analysis import rebuild_lowe_population as B
except ModuleNotFoundError:
    import rebuild_lowe_population as B

NS = {"c": "http://www.xml-cml.org/schema", "dl": "http://bitbucket.org/dan2097"}
SPACE = re.compile(r"\s+")
csv.field_size_limit(2**31 - 1)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def digest(text):
    return hashlib.sha1(SPACE.sub(" ", text).strip().encode()).hexdigest()


def reaction_key(reaction):
    r, g, p, u = B.roles(reaction)
    canonical = [B.canonical_components(r), B.canonical_components(g), B.canonical_components(p)]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def first(node, xpath):
    found = node.find(xpath, NS)
    return found.text.strip() if found is not None and found.text else ""


def extract_cml(root, member, patent, paragraph, smiles, archive=None):
    base = root.resolve(); path = (root / member).resolve()
    if base not in path.parents:
        raise ValueError("CML index points outside the specified source directory")
    if not path.exists() and archive:
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(archive.resolve()), "-C", str(root.resolve()), member], check=True)
    with path.open("rb") as handle:
        for _, node in ET.iterparse(handle, events=("end",)):
            if node.tag != "{" + NS["c"] + "}reaction":
                continue
            if (first(node, "dl:source/dl:documentId") == patent
                    and first(node, "dl:source/dl:paragraphNum") == paragraph
                    and first(node, "dl:reactionSmiles") == smiles):
                title = (first(node, "c:productList/c:product/c:molecule/dl:nameResolved")
                         or first(node, "dl:source/dl:headingText")
                         or first(node, "c:productList/c:product/c:molecule/c:name"))
                return {"title": title, "paragraph": first(node, "dl:source/dl:paragraphText"),
                        "cml_file_sha256": B.sha256(path)}
            node.clear()
    raise ValueError("Candidate CML reaction could not be found")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population-dir", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--cml-index", type=Path, required=True)
    p.add_argument("--cml-root", type=Path, required=True)
    p.add_argument("--cml-archive", type=Path, help="Extract missing source XML members to --cml-root without changing the random draw")
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--candidate-budget", type=int, default=300)
    p.add_argument("--eligibility-scope", choices=["count-criteria", "full-frame"], default="count-criteria",
                   help="Default: replace only records violating their single Table 1 count criterion")
    args = p.parse_args(argv)
    summary = json.loads((args.population_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary["complete_source"]:
        raise ValueError("Cannot draw using pilot template frequencies")
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark = read(args.benchmark); gt = read(args.ground_truth)
    if any(not r.get("patent_number", "").strip() for r in benchmark):
        raise ValueError("Every benchmark record requires patent_number to exclude previously sampled patents")
    used_patents = {r["patent_number"].strip() for r in benchmark}
    audit = read(args.population_dir / "benchmark_membership.csv")
    wanted = defaultdict(list)
    for r in audit:
        replace = (r.get("criteria", "") != r["assigned_stratum"] if args.eligibility_scope == "count-criteria"
                   else r["eligible"] != "True")
        if replace:
            wanted[r["assigned_stratum"]].append(int(r["idx"]))
    existing = {reaction_key(r["Lowe_smiles"]) for r in benchmark}
    texts = {digest(r["paragraph"]) for r in benchmark}
    pools = defaultdict(list)
    with gzip.open(args.population_dir / "sampling_frame.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["FrameStratum"] in wanted and r["DuplicateKey"] not in existing:
                pools[r["FrameStratum"]].append(r)
    candidates = []
    for stream, stratum in enumerate(B.LABELS):
        if stratum not in wanted:
            continue
        pool = pools[stratum]
        rng = np.random.default_rng([args.seed, stream])
        for rank, pos in enumerate(rng.permutation(len(pool))[:args.candidate_budget], 1):
            candidates.append({**pool[pos], "DrawRank": rank, "DrawStream": stream})
    bykey = defaultdict(list)
    for r in candidates:
        bykey[(r["PatentNumber"], r["ParagraphNum"], r["ReactionSmiles"])].append(r)
    # This optional index accelerates lookup only. Every accepted text is read
    # again from its original CML XML and its digest is checked below.
    with args.cml_index.open(encoding="utf-8-sig", newline="") as f:
        for link in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            key = (link["patent"], link["paragraph_num"], link["reaction_smiles"])
            for r in bykey.get(key, []):
                r.setdefault("links", []).append(link)
    selected, log = [], []
    used_text, used_reaction = set(texts), set(existing)
    for stratum in B.LABELS:
        ids = sorted(wanted.get(stratum, [])); accepted = 0
        for r in [x for x in candidates if x["FrameStratum"] == stratum]:
            if accepted >= len(ids):
                break
            reason, chosen = "no CML source link", None
            links = r.get("links", [])
            patent = r["PatentNumber"].strip()
            if not patent:
                reason, links = "missing patent identifier", []
            elif patent in used_patents:
                reason, links = "patent already in benchmark or selected replacements", []
            elif r["DuplicateKey"] in used_reaction:
                reason, links = "reaction already in benchmark or selected replacements", []
            for link in links:
                if not int(link["text_len"]):
                    reason = "empty paragraph"; continue
                if link["text_sha1"] in used_text:
                    reason = "paragraph already in benchmark"; continue
                found = extract_cml(args.cml_root, link["member"], r["PatentNumber"], r["ParagraphNum"], r["ReactionSmiles"], args.cml_archive)
                if digest(found["paragraph"]) != link["text_sha1"]:
                    raise ValueError("CML text digest differs from the supplied index")
                if not found["title"]:
                    reason = "missing title"; continue
                chosen = {**r, **found, "member": link["member"], "paragraph_sha1": link["text_sha1"]}
                reason = ""; break
            entry = {"stratum": stratum, "rank": r["DrawRank"], "raw_row": r["RawRow"],
                     "patent": r["PatentNumber"], "rejection": reason, "replaces_idx": ""}
            if chosen:
                idx = ids[accepted]; accepted += 1
                chosen["replaces_idx"] = idx; chosen.pop("links", None)
                selected.append(chosen); entry["replaces_idx"] = idx
                used_text.add(chosen["paragraph_sha1"]); used_reaction.add(chosen["DuplicateKey"])
                used_patents.add(patent)
            log.append(entry)
        if accepted != len(ids):
            write(args.output / "candidate_log.csv", log)
            raise ValueError(f"Increase --candidate-budget: {stratum} accepted {accepted}/{len(ids)}")
    selected.sort(key=lambda r: r["replaces_idx"])
    inputs = [{"idx": r["replaces_idx"], "Issue": repr([B.LABELS[r["FrameStratum"]]]),
               "patent_number": r["PatentNumber"],
               "title": r["title"], "paragraph": r["paragraph"], "Lowe_smiles": r["ReactionSmiles"]} for r in selected]
    write(args.output / "replacement_input_with_ids.csv", inputs)
    write(args.output / "input_new_examples_format.csv", inputs, ["Issue", "title", "paragraph", "Lowe_smiles"])
    write(args.output / "qwen_input.csv", [{"idx": "NEW"+str(r["idx"]), **{k: r[k] for k in ["title", "paragraph"]}} for r in inputs])
    write(args.output / "input_new_v2.csv", [{"record_id": "NEW"+str(r["idx"]), "replaces_idx": r["idx"],
          "title": r["title"], "paragraph": r["paragraph"]} for r in inputs])
    write(args.output / "id_map.csv", [{"local_idx": i, "server_idx": "NEW"+str(r["idx"]), "benchmark_idx": r["idx"]} for i, r in enumerate(inputs)])
    write(args.output / "selected_sources.csv", selected)
    write(args.output / "candidate_log.csv", log)
    pending = []
    for r in selected:
        reactants, reagents, products, _ = B.roles(r["ReactionSmiles"])
        item = {k: "" for k in gt[0]}
        item.update(idx=r["replaces_idx"], title=r["title"], paragraph=r["paragraph"],
                    review_status="pending_author_review", review_notes="",
                    candidate_reactants=json.dumps(B.canonical_components(reactants)),
                    candidate_reagents=json.dumps(B.canonical_components(reagents)),
                    candidate_products=json.dumps(B.canonical_components(products)))
        pending.append(item)
    write(args.output / "ground_truth_review.csv", pending)
    current ={int(r["idx"]): dict(r) for r in benchmark}
    for r in inputs:
        current[int(r["idx"])].update(r)
    write(args.output / "benchmark_input_candidate_400.csv", [current[i] for i in sorted(current)],
          ["idx", "Issue", "title", "paragraph", "Lowe_smiles", "patent_number"])
    cards = []
    for r in pending:
        cards.append(f"<article><h2>Record {r['idx']}: {html.escape(r['title'])}</h2><p>{html.escape(r['paragraph'])}</p><details><summary>Source-derived candidates (not approved ground truth)</summary><pre>{html.escape(json.dumps({k:v for k,v in r.items() if k.startswith('candidate_')},indent=2))}</pre></details></article>")
    (args.output / "ground_truth_review.html").write_text("<!doctype html><meta charset='utf-8'><title>Replacement paragraph review</title><style>body{max-width:1000px;margin:40px auto;font:17px/1.6 sans-serif}article{border-top:1px solid #aaa;padding:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Replacement paragraph review</h1><p>Review and finalize the labels and molecular ground truth in ground_truth_review.csv. Source-derived candidates have not been approved.</p>"+"".join(cards),encoding="utf-8")
    manifest = {"status": "awaiting_author_ground_truth_and_model_outputs", "policy": B.POLICY,
        "eligibility_scope": args.eligibility_scope,
        "seed": args.seed, "numpy": np.__version__, "sampling": "uniform source rows within each required sampling group; reject patents, reactions and paragraph text already in the benchmark or selected replacements before evaluation",
        "replaced_ids": [r["idx"] for r in inputs], "replacement_count": len(inputs),
        "old_benchmark_sha256": B.sha256(args.benchmark), "population_manifest_sha256": B.sha256(args.population_dir / "manifest.json"),
        "cml_index_sha256": B.sha256(args.cml_index), "cml_root": str(args.cml_root),
        "cml_archive_sha256": B.sha256(args.cml_archive) if args.cml_archive else None,
        "population_extrapolation_status": "not approved: retained historical rows were not sampled from the expanded new frame",
        "ground_truth_status": "pending author review; source-derived candidates are not labels",
        "qwen_status": "server input prepared; six configurations required", "performance_status": "not recomputed",
        "output_sha256": {f.name: B.sha256(f) for f in sorted(args.output.iterdir()) if f.is_file() and f.name != "manifest.json"}}
    B.dump(args.output / "manifest.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k != "output_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
