# Ovarian Cancer Immune Repertoire Analysis

This project involves the analysis of immune repertoire data obtained from the blood samples of healthy donors and ovarian cancer patients. 
The immune repertoire data includes T-cell receptor alpha (TRA) and beta (TRB) sequences obtained through Repertoire Sequencing (Rep-Seq). 
The pipeline covers data preprocessing, feature filtering, and machine learning for classification.

## Publication

**Zuckerbrot-Schuldenfrei, M.**, *et al.*
**"Ovarian cancer is detectable from peripheral blood using machine learning over T-cell receptor repertoires"**
*Briefings in Bioinformatics*, 2024.
[https://doi.org/10.1093/bib/bbae075](https://doi.org/10.1093/bib/bbae075)

## Data Availability
- The Rep-Seq data were deposited in the NCBI BioProject database under accession number [PRJNA1152888](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1152888).

---

## Repository Structure

This repository contains two versions of the analysis code:

```
Ovarian/
├── published/       # Code exactly as used in the paper — preserved for reproducibility
└── refactored/      # Engineering-grade rewrite: modular, reproducible, production-oriented
```

### `published/`
The original analysis code that produced the results reported in the paper. It is preserved exactly as run, with no modifications, so that the published results can be independently reproduced. See [`published/README.md`](published/README.md) for details.

### `refactored/`
A full rewrite of the pipeline applying software engineering best practices: a shared utility module, a pipeline orchestrator, and an inference script for new samples. The scientific logic is identical to the published version, only the code structure has changed. See [`refactored/README.md`](refactored/README.md) for details.

---

## Background

### Data Collection and Processing

1. **Blood Collection:**
   - Blood samples were collected from healthy donors and ovarian cancer patients.

2. **Repertoire Sequencing (Rep-Seq):**
   - Repertoire Sequencing was performed on the collected blood samples to obtain TRA and TRB sequences.

3. **Data Processing with MiXCR:**
   - The obtained raw sequencing data went through the MiXCR pipeline for processing.

4. **Concatenation of TRA and TRB Files:**
   - The processed TRA and TRB files were concatenated for further analysis.


### Analysis Overview

The pipeline runs in four stages:

1. **Immunarch analysis** — subsampling and exploratory analysis of TCR repertoires (clonality, diversity, overlap)
2. **Feature filtering** — two parallel feature engineering strategies:
   - *Method 1:* Top 600 most frequently shared clonotypes across samples
   - *Method 2:* 16 disease-associated clonotypes derived from VDJdb, McPAS, and TCGA-TRUST4 databases
3. **ML pipeline (600 features)** — SFM + SFS feature selection, multi-model comparison, winning model: GBM on 3 features
4. **ML pipeline (16 features)** — SFS feature selection, multi-model comparison, winning model: LDA on 4 features