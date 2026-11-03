"""Collapse predict_csv.py output (one row per paragraph x product) to one row per paragraph.

Per paragraph:
  * rows with a found product take priority over the empty "no product" row
  * primary product = the product whose role query extracted the most entities
    (ties -> the one mentioned last, usually the final compound in patent procedures)
  * role columns (schema tags only) are filled from the primary product;
    every product's roles are kept in all_roles_json

Usage: python collapse_predictions.py <predictions.csv> [--input ../input.csv] [--out ...]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROLE_TAGS = ['Reactants', 'Catalyst_Reagents', 'Solvent', 'Temperature', 'Time', 'Yield', 'Reaction']

p = argparse.ArgumentParser()
p.add_argument("pred_csv")
p.add_argument("--input", default=str(Path(__file__).resolve().parent.parent / "input.csv"))
p.add_argument("--id_col", default="idx")
p.add_argument("--out", default=None)
args = p.parse_args()

pred = pd.read_csv(args.pred_csv)
src = pd.read_csv(args.input, encoding="utf-8-sig")

rows = []
for pid, g in pred.groupby(args.id_col, sort=False):
    found = g[g["product"].notna()]
    base = {args.id_col: pid}
    if found.empty:
        rows.append({**base, "product_found": False, "n_products": 0, "primary_product": None,
                     **{t: None for t in ROLE_TAGS}, "all_products": "[]", "all_roles_json": "[]",
                     "prod_output": g["prod_output"].iloc[0]})
        continue
    cands = []
    for pos, (_, r) in enumerate(found.iterrows()):
        roles = json.loads(r["roles_json"]) if isinstance(r["roles_json"], str) else {}
        n = sum(len(roles.get(t, [])) for t in ROLE_TAGS)
        cands.append((n, pos, r["product"], roles, r["role_output"]))
    n, _, primary, roles, role_output = max(cands)
    rows.append({
        **base,
        "product_found": True,
        "n_products": len(cands),
        "primary_product": primary,
        **{t: json.dumps(roles.get(t, []), ensure_ascii=False) for t in ROLE_TAGS},
        "all_products": json.dumps([c[2] for c in cands], ensure_ascii=False),
        "all_roles_json": json.dumps([{"product": c[2], "roles": c[3]} for c in cands], ensure_ascii=False),
        "prod_output": found["prod_output"].iloc[0],
        "role_output": role_output,
    })

out_df = pd.DataFrame(rows)
# keep input.csv order and its original columns next to the predictions
merged = src.merge(out_df, on=args.id_col, how="left", validate="one_to_one")
out = Path(args.out) if args.out else Path(args.pred_csv).with_name(Path(args.pred_csv).stem + "_per_paragraph.csv")
merged.to_csv(out, index=False, encoding="utf-8-sig")
print(f"{len(pred)} rows -> {len(merged)} paragraphs ({int(merged['product_found'].sum())} with product) -> {out}")
