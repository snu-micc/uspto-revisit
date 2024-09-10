# uspto-revisit
Revisting the USPTO reaction dataset originally collected by Daniel Lowe using LLM (Gemini)<br>

## Contents

- [To do](#to-do)
- [Developer](#developer)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Data](#data)
- [Scripts](#scripts)
- [Publication](#publication)
- [References](#references)
- [License](#license)

## To do
1. Randomly select 100 reactions for each "Many products", "Many reagents", "Many reactants", "No reagent", "Rare template"<br>
: This is already selected and stored in 
2. Get structured summarization via LLMs
3. Get SMILES from various DB: OPSIN, PubChem, CIR, ChemSpider
4. Generate Reactions SMILES

## Developer
Shuan Chen (shuan75@snu.ac.kr)<br>
Chaewon Lee (cw.lee@snu.ac.kr)<br>

## Python Dependencies
* Python (version >= 3.8)
* OpenAI (version <= 2024)
* RDKit (version >= 2019)

## Installation Guide

```
git clone https://github.com/snu-micc/uspto-revisit.git
cd uspto-revisit
conda create -c conda-forge -n rdenv python -y
pip install google-generativeai
conda activate rdenv
```

## Data
#### USPTO dataset
The raw sgemented paragraphs and reactions extracted from USPTO are downloaded from this [Figshare link](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873).


## Scripts
See `gemini.ipynb`.

## Publication
-

## References
### Daniel Lowe's methods:
Phd thesis: https://www.repository.cam.ac.uk/items/dbb4f258-8f3c-4b59-9b5c-62fac7ca8c28 <br>
NextMove talks: https://www.nextmovesoftware.com/talks.html

### Prompt-based methods:
Omar M. Yaghi's group: https://pubs.acs.org/doi/10.1021/jacs.3c05819<br>
Eunomia: https://arxiv.org/pdf/2312.11690.pdf

### Fine-tuned-based methods:
Reaction Miner: https://aclanthology.org/2023.emnlp-demo.36/<br>

## License
This project is covered under the **MIT License**.
