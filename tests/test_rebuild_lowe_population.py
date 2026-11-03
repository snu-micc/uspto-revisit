"""Scientific invariants of the raw-source population builder (no model calls)."""
import csv
import gzip
import importlib.metadata
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from analysis import rebuild_lowe_population as B


class PopulationTests(unittest.TestCase):
    def test_atom_map_overlap_not_empty_agent_or_leaving_fragment(self):
        r, g, p, u = B.roles("[CH3:1]I.[NH3:2]>O>[CH3:1][NH2:2]")
        self.assertEqual((len(r), len(g), len(p)), (2, 1, 1))
        self.assertEqual(g, ["O"])
        self.assertEqual(B.criteria(2, 1, 1), [])
        self.assertEqual(B.criteria(3, 0, 2), ["many_products", "many_reactants", "no_reagent"])
        self.assertEqual(B.roles("[CH3:1]I.[NH3:2]>>[CH3:1][NH2:2].O")[3], ["O"])

    def test_duplicate_identity_ignores_atom_maps_and_molecule_order(self):
        a = B.canonical_components(["[CH3:1]I", "[NH3:2]"])
        b = B.canonical_components(["[NH3:99]", "I[CH3:21]"])
        self.assertEqual(a, b)
        self.assertNotEqual(B.canonical_components(["F[C@H](Cl)Br"]),
                            B.canonical_components(["F[C@@H](Cl)Br"]))

    def test_template_is_independent_of_map_renumbering(self):
        try:
            extract = B.load_extractor()
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("requires the mapping extra (localmapper)")
        a = extract("[CH3:1]I.[NH3:2]>>[CH3:1][NH2:2]", B.SETTINGS)
        b = extract("[NH3:83].I[CH3:71]>>[CH3:71][NH2:83]", B.SETTINGS)
        self.assertTrue(a)
        self.assertEqual(a, b)

    def test_duplicate_identity_includes_reagents_but_not_patent_or_paragraph(self):
        def record(reaction, patent="US1", paragraph="1"):
            source = {"ReactionSmiles": reaction, "PatentNumber": patent, "ParagraphNum": paragraph}
            return B.process_one((0, source))["DuplicateKey"]

        with patch.object(B, "load_extractor", return_value=lambda *args: "[C:1]>>[C:1]"):
            original = record("[CH3:1]I.[NH3:2]>O.CCO>[CH3:1][NH2:2]")
            self.assertTrue(original)
            self.assertEqual(original, record(
                "[NH3:83].I[CH3:71]>OCC.O>[CH3:71][NH2:83]", "US2", "99"))
            self.assertNotEqual(original, record("[CH3:1]I.[NH3:2]>O.CO>[CH3:1][NH2:2]"))
            self.assertNotEqual(original, record("[CH3:1]I.[NH3:2]>O.CCO.O>[CH3:1][NH2:2]"))
            self.assertNotEqual(original, record("[CH3:1]Br.[NH3:2]>O.CCO>[CH3:1][NH2:2]"))
            self.assertNotEqual(original, record("[CH3:1]I.[NH3:2]>O.CCO>[CH3:1][NH:2]C"))

    def test_full_corpus_frequency_duplicate_order_and_failed_records(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); (out / "chunks").mkdir()
            rows = []
            for i, (template, key, status) in enumerate([
                    ("T1", "same", "ok"), ("T1", "same", "ok"),
                    ("T2", "unique", "ok"), ("", "failed", "template_error")]):
                r = {k: "" for k in B.STAGE_FIELDS}
                r.update(RawRow=i, NReactants=2, NReagents=0, NProducts=1,
                         ReactionSmiles="C>>" + ["C", "N", "O", "F"][i], Template=template, DuplicateKey=key,
                         ProcessingStatus=status)
                rows.append(r)
            with gzip.open(out / "chunks/0000000.csv.gz", "wt", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=B.STAGE_FIELDS); w.writeheader(); w.writerows(rows)
            benchmark = out / "benchmark.csv"
            with benchmark.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["idx", "Lowe_smiles"])
                w.writeheader()
                w.writerows({"idx": i, "Lowe_smiles": r["ReactionSmiles"]}
                            for i, r in enumerate(rows))
            metadata = out / "benchmark_sample_membership.csv"
            with metadata.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["idx", "stratum"]); w.writeheader()
                w.writerows({"idx": i, "stratum": "no_reagent"} for i in reversed(range(len(rows))))
            B.finalize(out, benchmark, {"limit": 4}, metadata)
            with (out / "grants_issue.csv").open(encoding="utf-8") as handle:
                data = list(csv.DictReader(handle))
            self.assertEqual(data[0]["TemplateFreq"], "2")
            self.assertEqual(data[0]["FrameStratum"], "no_reagent")
            self.assertEqual(data[1]["DuplicateOfRawRow"], "0")
            self.assertEqual(data[1]["FrameStratum"], "no_reagent")
            self.assertIn("rare_temp", data[2]["Issue"])
            self.assertEqual(data[3]["TemplateFreq"], "")
            self.assertNotIn("rare_temp", data[3]["Issue"])
            self.assertEqual(data[3]["FrameStratum"], "no_reagent")
            with (out / "benchmark_membership.csv").open(encoding="utf-8") as handle:
                membership = list(csv.DictReader(handle))
            self.assertEqual([r["idx"] for r in membership if r["eligible"] == "False"], ["2"])
            self.assertEqual(membership[1]["reasons"], "")
            self.assertEqual(membership[3]["reasons"], "")
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["source_rows"], 4)
            self.assertEqual(summary["table1"]["no_reagent"], 4)
            self.assertEqual(summary["sampling_frame_total"], 3)
            self.assertFalse(summary["complete_source"])


if __name__ == "__main__":
    unittest.main()
