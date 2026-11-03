"""Two-stage extraction on unlabeled paragraphs (ChemRxnExtractor-style pipeline):

  1) Prod adapter : "annotate the products in the paragraph. <text>"      -> <Prod*..*Prod> tags
  2) Role adapter : one query per predicted product, with only that product tagged
                    "annotate the reaction roles in the paragraph. <text>" -> all role tags

Prompts are built by common.prompt_text, i.e. exactly the training template.

Usage:
  python predict_csv.py --model ../models/llama3-8b \
      --prod_adapter ../outputs/prod/<run_dir> --role_adapter ../outputs/role/<run_dir> \
      --csv ../input.csv --text_col paragraph --id_col idx
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from common import ROOT, TASKS, prompt_text

TAG_RE = re.compile(r"<(\w+)\*(.+?)\*\1>")


def best_checkpoint(path):
    path = Path(path)
    summary = path / "train_summary.json"
    return Path(json.loads(summary.read_text())["best_checkpoint"]) if summary.exists() else path


def pretokenize(text):
    """Approximate the space-separated style of the training data: '( 1.0 mmol )', '67 %', 'x , y'.
    Conservative: only splits punctuation at token boundaries, so names like
    '(+)-tartaric' or '3,4-dihydroxy' stay intact."""
    text = re.sub(r"\s+", " ", text.strip())
    out = []
    for tok in text.split(" "):
        pre, post = [], []
        dot = ""
        if len(tok) > 2 and tok[-1] == "." and tok[-2] in ")]":  # "ml)." -> "ml )." keeping the final period
            dot, tok = ".", tok[:-1]
        # split a leading bracket only if it is unbalanced within the token, so that
        # "(S)-3-...", "(R,R)-di-...", "(+)-tartaric" stay intact but "(350" -> "( 350"
        while len(tok) > 1 and ((tok[0] == "(" and tok.count("(") > tok.count(")")) or
                                (tok[0] == "[" and tok.count("[") > tok.count("]"))):
            pre.append(tok[0]); tok = tok[1:]
        while len(tok) > 1 and tok[-1] in ",;:)]" and (tok[-1] not in ")]" or tok.count("(") < tok.count(")") or tok.count("[") < tok.count("]")):
            post.insert(0, tok[-1]); tok = tok[:-1]
        tok = re.sub(r"^(\d+(?:\.\d+)?)%$", r"\1 %", tok)
        if dot:
            post[-1:] = [post[-1] + dot] if post else []
            tok = tok if post else tok + dot
        out.extend(pre + [tok] + post)
    return " ".join(out)


def strip_tags(text):
    return TAG_RE.sub(lambda m: m.group(2), text)


def find_span(plain, entity, occurrence):
    """(start, end) of the `occurrence`-th match of `entity` in plain text, or None.
    Falls back to a whitespace-insensitive match without trailing punctuation, since the
    model sometimes re-spaces names or appends a sentence-final period (as in its training data)."""
    matches = [(m.start(), m.end()) for m in re.finditer(re.escape(entity), plain)]
    if not matches:
        core = entity.rstrip(".,;").replace(" ", "")
        if not core:
            return None
        pattern = r"\s?".join(re.escape(c) for c in core)
        matches = [(m.start(), m.end()) for m in re.finditer(pattern, plain)]
    if not matches:
        return None
    return matches[min(occurrence, len(matches) - 1)]


def tag_span(plain, s, e):
    return f"{plain[:s]}<Prod*{plain[s:e]}*Prod>{plain[e:]}"


# a period ends a sentence only if followed by a capital/digit/bracket and not after a common abbreviation
_ABBREV = r"(?<!\bM\.p)(?<!\bconc)(?<!\bca)(?<!\bapprox)(?<!\be\.g)(?<!\bi\.e)(?<!\bNo)(?<!\bFig)(?<!\bref)(?<!\bEx)"
_SENT_END = re.compile(_ABBREV + r"\.\s+(?=[A-Z0-9(\[])")


def split_sentences(plain):
    """[(offset, sentence)] covering `plain`; the period stays with its sentence."""
    out, start = [], 0
    for m in _SENT_END.finditer(plain):
        out.append((start, plain[start:m.start() + 1]))
        start = m.end()
    out.append((start, plain[start:]))
    return [(o, t) for o, t in out if t.strip()]


def products_with_occurrence(tagged):
    """[(entity, k)] where the product is the k-th occurrence of `entity` in the untagged text."""
    res = []
    for m in TAG_RE.finditer(tagged):
        if m.group(1) != "Prod":
            continue
        prefix = strip_tags(tagged[:m.start()])
        ent = m.group(2)
        res.append((ent, prefix.count(ent)))
    return res


def entities(tagged):
    d = {}
    for m in TAG_RE.finditer(tagged):
        d.setdefault(m.group(1), []).append(m.group(2))
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prod_adapter", required=True)
    p.add_argument("--role_adapter", required=True)
    p.add_argument("--csv", default=str(ROOT / "input.csv"))
    p.add_argument("--text_col", default="paragraph")
    p.add_argument("--id_col", default="idx")
    p.add_argument("--no_pretokenize", action="store_true")
    p.add_argument("--prod_unit", choices=["sentence", "paragraph"], default="paragraph",
                   help="input unit for the product stage; sentence found fewer products on input.csv (134 vs 187 paragraphs, job 6286 vs 6285)")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_mem", type=float, default=0.85)
    p.add_argument("--out", default=str(ROOT / "outputs" / "input_predictions.csv"))
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    texts = [t if args.no_pretokenize else pretokenize(t) for t in df[args.text_col].astype(str)]

    llm = LLM(model=args.model, enable_lora=True, max_loras=2, max_lora_rank=args.lora_r,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem, seed=0)
    sampling = SamplingParams(temperature=0, top_p=1, max_tokens=3072, stop=["!!!"])
    prod_lora = LoRARequest("prod", 1, str(best_checkpoint(args.prod_adapter)))
    role_lora = LoRARequest("role", 2, str(best_checkpoint(args.role_adapter)))

    # Stage 1: products, per sentence (or per paragraph)
    units = []  # (row index, char offset in paragraph, text)
    for i, plain in enumerate(texts):
        units += [(i, o, t) for o, t in (split_sentences(plain) if args.prod_unit == "sentence" else [(0, plain)])]
    prod_out = llm.generate([prompt_text(TASKS["prod"]["instruction"], u[2]) for u in units], sampling, lora_request=prod_lora)
    prod_tagged = [[] for _ in texts]
    spans = [[] for _ in texts]  # product spans in paragraph coordinates, in text order
    for (i, off, unit), o in zip(units, prod_out):
        tagged = o.outputs[0].text.strip()
        prod_tagged[i].append(tagged)
        for ent, k in products_with_occurrence(tagged):
            span = find_span(unit, ent, k)
            # span None: product string not found in the input -> hallucinated, skip
            if span is not None and (off + span[0], off + span[1]) not in spans[i]:
                spans[i].append((off + span[0], off + span[1]))
    prod_tagged = [" ".join(t) for t in prod_tagged]

    # Stage 2: one role query per product, on the whole paragraph
    queries = []  # (row index, product, role input)
    for i, plain in enumerate(texts):
        for st, en in sorted(spans[i]):
            queries.append((i, plain[st:en], tag_span(plain, st, en)))
    role_out = llm.generate([prompt_text(TASKS["role"]["instruction"], q[2]) for q in queries], sampling, lora_request=role_lora)

    rows = []
    by_row = {}
    for (i, ent, role_in), o in zip(queries, role_out):
        by_row.setdefault(i, []).append((ent, role_in, o.outputs[0].text.strip()))
    for i in range(len(df)):
        base = {args.id_col: df[args.id_col].iloc[i], "input_text": texts[i], "prod_output": prod_tagged[i]}
        if i not in by_row:
            rows.append({**base, "product": None, "role_input": None, "role_output": None, "roles_json": None})
        for ent, role_in, role_txt in by_row.get(i, []):
            rows.append({**base, "product": ent, "role_input": role_in, "role_output": role_txt,
                         "roles_json": json.dumps(entities(role_txt), ensure_ascii=False)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    n_prod = sum(1 for r in rows if r["product"])
    print(f"{len(df)} paragraphs -> {n_prod} (paragraph, product) rows; {len(df) - len(by_row)} paragraphs with no product -> {out}")


if __name__ == "__main__":
    main()
