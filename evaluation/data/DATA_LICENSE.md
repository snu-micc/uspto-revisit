# Data artifact license and attribution

## License

The separately distributed `grants_issue.zip` data artifact is licensed under
the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0). This license applies to the data artifact, not automatically to
the software in this repository.

## Attribution

The artifact is an adaptation of patent-derived reaction records from the Open
Reaction Database `uspto-grants` dataset. The upstream ORD datasets are
distributed under
[CC BY-SA 4.0](https://github.com/open-reaction-database/ord-data/blob/main/LICENSE).
The selection uses criteria associated with ORDerly, whose software is
MIT-licensed and whose method is described by Wigh et al., *J. Chem. Inf.
Model.* (2024), [DOI: 10.1021/acs.jcim.4c00292](https://doi.org/10.1021/acs.jcim.4c00292).
Please also cite the present study when using this released artifact.

## Changes from the upstream data

The released archive contains the 623,290 reaction records selected by the
study's ORDerly issue criteria, represented as one flattened CSV. It includes
the study's stored `Template`, `Reagent`, `NAtomChanges`, `TemplateFreq`,
`NMolecules`, and `Issue` analysis fields. It does not include original patent
full text. The historical workflow that generated the stored template and
frequency fields is not included, so this release supports downstream linkage
and threshold-sensitivity reproduction rather than independent regeneration of
those fields from the full corpus.
