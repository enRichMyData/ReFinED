# Challenge Datasets — HTR1 & HTR2 (SemTab)

This ZIP contains the two **SemTab (Table-to-KG) challenge datasets**, each being a
**dataset (a collection of many tables)** used to benchmark entity/column linking:

- **HTR1** — Hard Table Retrieval 1  
- **HTR2** — Hard Table Retrieval 2

## Contents & Layout
Each challenge **dataset** folder typically includes:
- `tables/` — **multiple** input CSV tables
- `gt/` — ground-truth CSV(s) with the expected Wikidata QIDs (by cell/mention)
- `mention_to_qid/` — JSON mapping(s) from surface mention → candidate QIDs
- Optional JSONs in the dataset root:
  - `cell_to_qid.json`
  - `column_classifications.json`

## Conventions
- String similarity statistics are computed on **lowercased** text.
- Cell IDs follow the unified format: `tableName-idRow-idCol` (`idRow` is 0-indexed).
- Evaluators must score **only** cells present in the GT (ignore extra predictions).

## Licensing
- **HTR1 & HTR2:** CC BY 4.0.

## How to Cite
If you use these resources, please **cite both**:
1. **This Zenodo record** (v2):  
   Avogadro, R., & Rauniyar, A. (2025). *Benchmark Datasets for Entity Linking from Tabular Data (Version 2).* Zenodo. https://doi.org/10.5281/zenodo.15888942
2. The **SemTab Challenge** (Table-to-KG Matching).  
   *SemTab: Semantic Web Challenge on Tabular Data to Knowledge Graph Matching (Table-to-KG).*  
   (Please reference the SemTab challenge overview appropriate to your year of use.)