import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from analysis import prepare_lowe_replacements as P


class ReplacementTests(unittest.TestCase):
    def test_only_ineligible_record_changes_and_new_labels_remain_unapproved(self):
        with tempfile.TemporaryDirectory() as td:
            w = Path(td); pop = w / "population"; pop.mkdir(); cml = w / "cml"; cml.mkdir()
            (pop / "summary.json").write_text(json.dumps({"complete_source": True}))
            (pop / "manifest.json").write_text('{}')
            old = [
                {"idx": "0", "patent_number": "USOLD0", "stratum": "many_products", "Issue": "['many_product (> 1)']", "title": "old zero", "paragraph": "Old text zero", "Lowe_smiles": "[CH3:1]I.[NH3:2]>>[CH3:1][NH2:2]"},
                {"idx": "1", "patent_number": "USOLD1", "stratum": "no_reagent", "Issue": "['no_reagent']", "title": "keep one", "paragraph": "Keep text one", "Lowe_smiles": "[CH3:1][CH2:2]I.[NH3:3]>>[CH3:1][CH2:2][NH2:3]"}]
            P.write(w / "benchmark.csv", old)
            P.write(w / "gt.csv", [{"idx": "0", "title": "old zero", "paragraph": "Old text zero", "label": "noise-free", "ground_truth_reaction_smiles": "approved old GT"}])
            # A newly assigned rarity/duplicate exclusion must not trigger an
            # extra replacement when the user requested count criteria only.
            P.write(pop / "benchmark_membership.csv", [{"idx": "0", "assigned_stratum": "many_products", "criteria": "many_products+no_reagent", "eligible": "False"}, {"idx": "1", "assigned_stratum": "no_reagent", "criteria": "no_reagent", "eligible": "False"}])
            reaction = "[CH3:3][OH:4].[Na+:5]>O>[CH2:3]=[O:4].[Na+:5]"
            record = {"RawRow": "10", "PatentNumber": "USX", "ParagraphNum": "7", "ReactionSmiles": reaction,
                      "FrameStratum": "many_products", "DuplicateKey": P.reaction_key(reaction)}
            with gzip.open(pop / "sampling_frame.csv.gz", "wt", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(record)); writer.writeheader(); writer.writerow(record)
            text = "A new source paragraph with an oxidation."
            xml = f'<cml xmlns="{P.NS["c"]}" xmlns:dl="{P.NS["dl"]}"><reaction><productList><product><molecule><dl:nameResolved>New title</dl:nameResolved></molecule></product></productList><dl:reactionSmiles>{escape(reaction)}</dl:reactionSmiles><dl:source><dl:documentId>USX</dl:documentId><dl:paragraphNum>7</dl:paragraphNum><dl:paragraphText>{text}</dl:paragraphText></dl:source></reaction></cml>'
            (cml / "one.xml").write_text(xml)
            with (w / "index.tsv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, delimiter="\t", fieldnames=["member", "patent", "paragraph_num", "reaction_smiles", "text_sha1", "text_len"])
                writer.writeheader(); writer.writerow({"member": "one.xml", "patent": "USX", "paragraph_num": "7", "reaction_smiles": reaction, "text_sha1": P.digest(text), "text_len": len(text)})
            P.main(["--population-dir", str(pop), "--benchmark", str(w / "benchmark.csv"),
                    "--ground-truth", str(w / "gt.csv"), "--cml-index", str(w / "index.tsv"),
                    "--cml-root", str(cml), "--output", str(w / "out")])
            new = P.read(w / "out/benchmark_input_candidate_400.csv")
            self.assertEqual(new[0]["title"], "New title")
            self.assertEqual(new[1]["paragraph"], old[1]["paragraph"])
            self.assertEqual(new[1]["Lowe_smiles"], old[1]["Lowe_smiles"])
            labels = P.read(w / "out/ground_truth_review.csv")
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0]["label"], "")
            self.assertEqual(labels[0]["review_status"], "pending_author_review")
            self.assertEqual(P.read(w / "out/id_map.csv")[0]["benchmark_idx"], "0")


@pytest.fixture
def draw_case(tmp_path, monkeypatch):
    """Two vacancies and three candidates, inspected in a fixed order."""
    pop = tmp_path / "population"; pop.mkdir()
    cml = tmp_path / "cml"; cml.mkdir()
    (pop / "summary.json").write_text(json.dumps({"complete_source": True}))
    (pop / "manifest.json").write_text("{}")
    old = [{"idx": str(i), "patent_number": f"USOLD{i}", "stratum": "many_products",
            "Issue": "['many_product (> 1)']", "title": f"Old {i}",
            "paragraph": f"Old text {i}", "Lowe_smiles": "C" * (i + 1) + ">>O"}
           for i in range(2)]
    P.write(tmp_path / "benchmark.csv", old)
    P.write(tmp_path / "gt.csv", [{"idx": "0", "label": "noise-free"}])
    P.write(pop / "benchmark_membership.csv", [
        {"idx": str(i), "assigned_stratum": "many_products",
         "criteria": "many_products+no_reagent", "eligible": "False"} for i in range(2)])

    class FixedOrder:
        def permutation(self, n):
            return P.np.arange(n)

    monkeypatch.setattr(P.np.random, "default_rng", lambda seed: FixedOrder())
    args = ["--population-dir", str(pop), "--benchmark", str(tmp_path / "benchmark.csv"),
            "--ground-truth", str(tmp_path / "gt.csv"), "--cml-index", str(tmp_path / "index.tsv"),
            "--cml-root", str(cml), "--output", str(tmp_path / "out")]

    def prepare(kind):
        records, links = [], []
        for i in range(3):
            patent = "USNEW0" if i == 1 and kind == "new_patent" else f"USNEW{i}"
            if i == 1 and kind == "old_patent": patent = "USOLD1"
            chain = 0 if i == 1 and kind == "new_reaction" else i
            reaction = "C" * (chain + 1) + "[CH:1]=[O:2]>O>" + "C" * (chain + 1) + "[CH2:1][OH:2]"
            text = "New paragraph 0" if i == 1 and kind == "new_text" else f"New paragraph {i}"
            records.append({"RawRow": str(10 + i), "PatentNumber": patent, "ParagraphNum": str(i),
                            "ReactionSmiles": reaction, "FrameStratum": "many_products",
                            "DuplicateKey": P.reaction_key(reaction)})
            xml = (f'<cml xmlns="{P.NS["c"]}" xmlns:dl="{P.NS["dl"]}"><reaction>'
                   f'<productList><product><molecule><dl:nameResolved>New {i}</dl:nameResolved>'
                   f'</molecule></product></productList><dl:reactionSmiles>{escape(reaction)}</dl:reactionSmiles>'
                   f'<dl:source><dl:documentId>{patent}</dl:documentId><dl:paragraphNum>{i}</dl:paragraphNum>'
                   f'<dl:paragraphText>{text}</dl:paragraphText></dl:source></reaction></cml>')
            (cml / f"{i}.xml").write_text(xml)
            links.append({"member": f"{i}.xml", "patent": patent, "paragraph_num": str(i),
                          "reaction_smiles": reaction, "text_sha1": P.digest(text), "text_len": len(text)})
        with gzip.open(pop / "sampling_frame.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        with (tmp_path / "index.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(links[0]), delimiter="\t"); writer.writeheader(); writer.writerows(links)
        return args

    return prepare


@pytest.mark.parametrize("kind,reason", [
    ("old_patent", "patent already"), ("new_patent", "patent already"),
    ("new_reaction", "reaction already"), ("new_text", "paragraph already"),
])
def test_draw_rejects_existing_and_new_duplicates(draw_case, tmp_path, kind, reason):
    P.main(draw_case(kind))
    selected = P.read(tmp_path / "out/selected_sources.csv")
    assert [r["RawRow"] for r in selected] == ["10", "12"]
    log = P.read(tmp_path / "out/candidate_log.csv")
    assert reason in log[1]["rejection"] and log[1]["replaces_idx"] == ""
    revised = P.read(tmp_path / "out/benchmark_input_candidate_400.csv")
    assert [r["patent_number"] for r in revised] == ["USNEW0", "USNEW2"]


def test_draw_requires_patent_identifiers(draw_case, tmp_path):
    args = draw_case("old_patent")
    rows = P.read(tmp_path / "benchmark.csv")
    rows[0]["patent_number"] = ""
    P.write(tmp_path / "benchmark.csv", rows)
    with pytest.raises(ValueError, match="requires patent_number"):
        P.main(args)


if __name__ == "__main__":
    unittest.main()
