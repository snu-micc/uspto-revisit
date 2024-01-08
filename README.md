# uspto-revisit
Revisting the USPTO reaction dataset originally collected by Daniel Lowe using LLM (Gemini)<br>

## Contents

- [To do](#to-do)
- [Developer](#developer)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Data](#data)
- [Demo](#demo)
- [Publication](#publication)
- [Reference](#reference)
- [License](#license)

## To do
1. Some reagents are either not extracted in reactants list or not show in reaction step
2. Some molecules cannot find their SMILES through PubChem
3. Decide what to show at for "mixture" product
4. Review the validity of extracted reactoin string compared to Lowe's data
5. Slow inference time

## Developer
Shuan Chen (shuan75@snu.ac.kr)<br>

## Python Dependencies
* Python (version >= 3.8)
* google-generativeai (version <= 2023)
* RDKit (version >= 2019)

## Installation Guide

```
git clone https://github.com/snu-micc/uspto-revisit.git
cd MechFinder
conda create -c conda-forge -n rdenv python -y
pip install google-generativeai
conda activate rdenv
```

## Data
#### USPTO dataset
The raw sgemented paragraphs and reactions extracted from USPTO are downloaded from this [Figshare link](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873).


## Demo
See `Demo.ipynb` for running instructions and expected output.

```
```

## Publication
Under review

## References
### Daniel Lowe's methods:
Phd thesis: https://www.repository.cam.ac.uk/items/dbb4f258-8f3c-4b59-9b5c-62fac7ca8c28
NextMove talks: https://www.nextmovesoftware.com/talks.html

### Prompt-based methods:
Omar M. Yaghi's group: https://pubs.acs.org/doi/10.1021/jacs.3c05819
Eunomia: https://arxiv.org/pdf/2312.11690.pdf

### Fine-tuned-based methods:
Reaction Miner: https://aclanthology.org/2023.emnlp-demo.36/

## License
This project is covered under the **The MIT License**.
