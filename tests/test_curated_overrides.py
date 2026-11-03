"""The secondary scenario may change only components named in its 46 rules."""
import json
from pathlib import Path

import pandas as pd

from analysis import curation_ablation as CA
from analysis.build_curated_outputs import load_rules, override_edits

ROOT=Path(__file__).resolve().parents[1]


def test_unlisted_aliases_and_row_review_names_are_not_applied():
    rules=load_rules()
    names={'Reactants, Solvents, Catalysts':{'A':'palladium on carbon','B':'DCM','C':'alcohol'}}
    frame=pd.DataFrame([{'prediction':json.dumps(names),
                         'm_smiles':json.dumps({'A':'[palladium on carbon (NoSmi)]','B':'CO','C':'[alcohol (NoSmi)]'})}])
    edits,audit=override_edits({'m':frame},rules)
    assert edits=={('m',0):{'A':'[Pd]'}}
    assert [a['name'] for a in audit]==['palladium on carbon']


def test_released_curated_outputs_are_primary_plus_only_the_46_rules():
    rules=load_rules()
    manifest=json.loads((ROOT/'result/model_outputs/curated/manifest.json').read_text(encoding='utf-8'))
    count=0
    for p in sorted((ROOT/'result/model_outputs/fully_automatic').glob('*.csv')):
        model=p.name.removesuffix('_reaction_smiles_final.csv')
        before=pd.read_csv(p,encoding='utf-8-sig',dtype=str,keep_default_na=False)
        after=pd.read_csv(ROOT/'result/model_outputs/curated'/p.name,encoding='utf-8-sig',dtype=str,keep_default_na=False)
        assert before.shape==after.shape
        derived={f'{model}_{c}' for c in ['smiles','rxn',*CA.MAPPING_COLUMNS]}
        fixed=[c for c in before if c not in derived]
        pd.testing.assert_frame_equal(before[fixed],after[fixed])
        for idx,row in before.iterrows():
            old=CA.loads_any(row[f'{model}_smiles'])
            new=CA.loads_any(after.at[idx,f'{model}_smiles'])
            assert old.keys()==new.keys()
            names=CA.prediction_names(row['prediction'])
            for label in old:
                if old[label]!=new[label]:
                    rule=rules[CA.normalize_name_key(names[label])]
                    assert new[label]==rule['resolved_smiles']
                    count+=1
    assert count==manifest['changed_components']>0
    assert manifest['override_count']==46
    assert manifest['historical_lookup_tables_used'] is False
    assert manifest['row_decisions_used'] is False
    assert manifest['ground_truth_read'] is False
