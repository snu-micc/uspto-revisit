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
- [License](#license)


## Workflow

### 1. Randomly Select 100 Reactions for Each Category
- **Categories**: "Many products", "Many reagents", "Many reactants", "No reagent", "Rare template".

### 2. Get Structured Summarization via LLMs
- Prompts for structured summarization are available in Supplementary Inforamtion of [#Journal link]
  
### 3. Get SMILES from Various Databases
- SMILES strings are retrieved from OPSIN, PubChem, CIR, and ChemSpider.
- Use your own API key for ChemSpider. You can obtain API key from the [ChemSpider API Link](https://developer.rsc.org/)
- Code to retrieve SMILES can be found in :  
  `smiles_fetch_utils.py`
  `chemspider_utils.py`
- Log for retreiving SMILES can be found in:
  `smiles_fetch.log`

### 4. Generate Reaction SMILES
- Reaction SMILES are generated based on the selected reactions.
- The code for generating reaction SMILES is located in:  
  `reaction_step_processing_utlis.py`
  `reaction_smiles_processing_utils.py`

## Developer
Chaewon Lee (cw.lee@snu.ac.kr)<br>

## Python Dependencies
*aiohttp==3.9.5
*ChemSpiPy==2.0.0
*nest_asyncio==1.6.0
*Requests==2.32.3


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
```bibtex
@article{chen2024improve,
  title={Improving Reaction Dataset Extraction with Fine-Tuned Large Language Models},
  author={Chaewon Lee, Shuan Chen, Kai Tzu-iunn Ong, Jinyoung Yeo, and Yousung Jung},
  journal={In review},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```


## License
This project is covered under the **MIT License**.

