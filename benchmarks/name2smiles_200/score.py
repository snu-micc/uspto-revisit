"""Score original and rerun predictions against the same corrected references, offline."""
from pathlib import Path
import argparse, csv, json, math
from rdkit import Chem, RDLogger, rdBase

HERE = Path(__file__).resolve().parent
MATCHING_RULE = "RDKit canonical isomeric SMILES equality; no tautomer normalization"


def canonical(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule is not None else None


def load_records(path):
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    indexed = {r["row_id"]: r for r in rows}
    if len(rows) != 200 or set(indexed) != set(range(1, 201)):
        raise ValueError(f"{path}: expected exactly one record for each compound ID 1-200")
    return indexed


def score_prediction(record, reference):
    smiles = record.get("smiles")
    value = canonical(smiles)
    matches = bool(reference and value and reference == value)
    if not record:
        status = "missing_record"
    elif record.get("error"):
        status = "generation_error"
    elif not isinstance(smiles, str) or not smiles.strip():
        status = "null_or_empty_prediction"
    elif value is None:
        status = "invalid_smiles"
    else:
        status = "valid_smiles"
    correct = status == "valid_smiles" and matches
    reason = (status if status != "valid_smiles" else
              "exact_match" if matches else "canonical_structure_mismatch")
    return {"prediction_smiles": smiles or "", "prediction_canonical_smiles": value or "",
            "prediction_status": status, "matches_corrected_reference": matches,
            "counted_as_correct": correct, "scoring_reason": reason}


def wilson(k, n):
    z = 1.959963984540054
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [100*(c-h), 100*(c+h)]


def summarize(rows, prefix):
    correct = sum(r[f"{prefix}_counted_as_correct"] for r in rows)
    statuses = [r[f"{prefix}_prediction_status"] for r in rows]
    return {"records_present": sum(s != "missing_record" for s in statuses),
            "correct": correct, "accuracy_percent": 100*correct/len(rows),
            "wilson95_percent": wilson(correct, len(rows)),
            "valid_smiles": sum(s == "valid_smiles" for s in statuses),
            "failure_counts": {s: statuses.count(s) for s in
                               ("missing_record", "generation_error", "null_or_empty_prediction", "invalid_smiles")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"results")
    parser.add_argument("--records-dir", type=Path,
                        help="Optional new-generation records for the rerun columns; original predictions stay fixed.")
    args = parser.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    refs = load_records(HERE/"references.json")
    original = load_records(HERE/"original_predictions.json")
    rerun = load_records(HERE/"predictions.json")
    names = load_records(HERE/"names_only.json")
    rows = []
    for i, ref in sorted(refs.items()):
        if original[i]["name"] != ref["name"] or names[i]["name"] != ref["name"]:
            raise ValueError(f"Compound {i}: chemical names are not aligned")
        reference = canonical(ref["reference_smiles"])
        if reference is None:
            raise ValueError(f"Compound {i}: invalid reference SMILES")
        path = args.records_dir/f"{i:03d}.json" if args.records_dir else None
        prediction = (json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}) if path else rerun[i]
        row = {"compound_id": i, "chemical_name": ref["name"],
               "corrected_reference_smiles": ref["reference_smiles"],
               "corrected_reference_canonical_smiles": reference,
               "matching_rule": MATCHING_RULE}
        for prefix, record in (("original", original[i]), ("rerun", prediction)):
            row.update({f"{prefix}_{k}": v for k, v in score_prediction(record, reference).items()})
        row.update(rerun_response_model_id=prediction.get("response_model") or "",
                   rerun_generation_error=prediction.get("error") or "")
        rows.append(row)
    summary = {"rdkit": rdBase.rdkitVersion, "n": len(rows), "matching_rule": MATCHING_RULE,
               "original": summarize(rows, "original"), "rerun": summarize(rows, "rerun"),
               "rerun_response_models": sorted({r["rerun_response_model_id"] for r in rows if r["rerun_response_model_id"]}),
               "interpretation": "Agreement with the same fixed author-reviewed references; not independent validation of reference correctness."}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
