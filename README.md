# USPTO-Revisit
Revisting the USPTO reaction dataset originally collected by Daniel Lowe using LLM (GPT4o-mini)<br>

## Contents

- [Workflow](#Workflow)
- [Developer](#developer)
- [Python Dependencies](#python-dependencies)
- [Installation Guide](#installation-guide)
- [Data](#data)
- [Scripts](#scripts)
- [Publication](#publication)
- [References](#references)
- [License](#license)


## Workflow

### 1. Randomly Select 100 Reactions for Each Category
- **Categories**: "Many products", "Many reagents", "Many reactants", "No reagent", "Rare template".

### 2. Get Structured Summarization via LLMs
- Prompts for structured summarization are available in [#Journal link]:
  
### 3. Get SMILES from Various Databases
- SMILES strings are retrieved from OPSIN, PubChem, CIR, and ChemSpider.
- Code to retrieve SMILES can be found in :  
  `smiles_fetch_utils.py`, `chemspider_utils.py`
- Log for retreiving SMILES can be found in:
  `smiles_fetch.log`

### 4. Generate Reaction SMILES
- Reaction SMILES are generated based on the selected reactions.
- The code for generating reaction SMILES is located in:  
  `reaction_step_processing_utlis.py`, `reaction_smiles_processing_utils.py`

## Developer
Shuan Chen (shuan75@snu.ac.kr)<br>
Chaewon Lee (cw.lee@snu.ac.kr)<br>

## Python Dependencies
* Python (version >= 3.8)
* OpenAI (version <= 2024)

## Installation Guide

```
git clone https://github.com/snu-micc/uspto-revisit.git
cd uspto-revisit
conda create -c conda-forge -n rdenv python -y
conda activate rdenv
```

## Data
#### USPTO dataset
The raw segemented paragraphs and reactions extracted from US patents are downloaded from this [Figshare link](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873).

## Scripts
See the `main.ipynb` for details on running the scripts.

## Publication
USTPO-multistep reaction dataset: [Figshare link](https://figshare.com/articles/dataset/USPTO-multistep_csv/26941993?file=49017574).


## References
### Daniel Lowe's methods:
Ph.D Thesis: [Respository link](https://www.repository.cam.ac.uk/items/dbb4f258-8f3c-4b59-9b5c-62fac7ca8c28)


## License
This project is covered under the **MIT License**.

