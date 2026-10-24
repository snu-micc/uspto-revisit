# Independent blinded label assessment

This release contains 45 current-benchmark paragraphs with binary judgments
from both evaluators. The analysis subset includes every eligible paired
judgment, including disagreements. It excludes replaced benchmark records
and records without two binary judgments.

`paired_labels.csv` contains only the zero-based `benchmark_id` and the
anonymous `evaluator_A` and `evaluator_B` labels (`noise-free` or `noisy`).
`benchmark_id` corresponds to `idx` in
[`../revised/benchmark_sample_membership.csv`](../revised/benchmark_sample_membership.csv)
and the zero-based row position, excluding the header, in
[`../revised/input.csv`](../revised/input.csv).
Evaluator names, free-text comments, login information and the private study
database are not included.

The evaluators were blinded to prior labels, ground-truth structures, model
outputs and each other's answers. They entered chemical names, used
OPSIN/PubChem conversion, reviewed structures and locked their submissions.
These results measure binary-label agreement before adjudication. They do
not constitute independently adjudicated ground truth or independently
re-scored model performance.

There are 37 agreements and 8 disagreements. Agreement is 82.2%
(95% Wilson interval 68.7–90.7%), with Cohen's kappa 0.604.
`summary.json` retains the unrounded values and the input-file SHA-256.

From the repository root, using Python 3.9 or newer and no extra packages:

```bash
python evaluation/reproduce_blind_agreement.py
```

To regenerate the recorded summary:

```bash
python evaluation/reproduce_blind_agreement.py --output evaluation/data/independent_assessment/summary.json
```
