# CAGE: A Causality-Aware, Reproducible Pipeline for Candidate Driver Gene Prioritization in Heterogeneous Tumor Cohorts

A reproducible, end-to-end computational pipeline for prioritising candidate
driver genes in heterogeneous tumour cohorts, with **esophageal carcinoma
(ESCA)** as the primary application. CAGE integrates patient-grouped
cross-validation, a sparse invariant adversarial autoencoder–classifier,
multi-component gene ranking (Causality-Aware Priority Score, CDPS), and
orthogonal biological validation across independent cohorts.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#requirements)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Backend: NumPy](https://img.shields.io/badge/backend-pure%20NumPy-orange.svg)](#design-philosophy)

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Obtaining Input Data](#obtaining-input-data)
   - [TCGA-ESCA RNA-seq (primary cohort)](#tcga-esca-rna-seq-primary-cohort)
   - [GEO external validation cohorts](#geo-external-validation-cohorts)
   - [Pathway databases (MSigDB)](#pathway-databases-msigdb)
   - [Optional: protein–protein interaction edge list](#optional-proteinprotein-interaction-edge-list)
6. [Repository Layout](#repository-layout)
7. [Pipeline Architecture](#pipeline-architecture)
8. [Quick Start](#quick-start)
9. [Module Reference](#module-reference)
10. [Global CLI Options](#global-cli-options)
11. [Figure and Visualization Standards](#figure-and-visualization-standards)
12. [Output Directory Structure](#output-directory-structure)
13. [Reproducibility](#reproducibility)
14. [Testing](#testing)
15. [Claim Boundaries](#claim-boundaries)
16. [Citation](#citation)
17. [License](#license)

---

## Overview

CAGE is a modular pipeline that takes raw RNA-seq counts and clinical metadata
as input and produces a fully documented set of prioritised candidate genes
together with publication-ready figures, tables, and manuscript drafts.
All numerical components — logistic regression, elastic net, decision trees,
random forests, and the sparse invariant deep model — are implemented from
first principles in pure NumPy.

### Key components

| Component | Description |
|-----------|-------------|
| **Sparse Feature Gate** | Learnable per-gene gate with L1 or hard-concrete regularisation for automatic feature selection. |
| **Confounder Adversary** | Gradient-reversal layer that removes histology, sex, and smoking signal from the latent space. |
| **Environment Invariance** | Latent-mean penalty that encourages disease representations to generalise across patient strata. |
| **CDPS** | Five-component score combining attribution, gate weight, stability, invariance, and in-silico perturbation. |
| **Biological Validation** | Differential expression, pathway enrichment, PPI network support, clinical association, and subgroup robustness. |
| **External Validation** | Direction-of-effect replication and classifier transfer across four independent GEO cohorts. |
| **Public Biological Validation** | Downstream HPA, cBioPortal, immune/TME marker, optional single-cell, and integrated manuscript-oriented validation for final candidates. |

---

## Design Philosophy

CAGE is intentionally built on a **pure-NumPy** numerical backend. All gradients
are derived analytically and every loss term is implemented explicitly. The
core modelling workflow does not depend on scikit-learn, PyTorch, TensorFlow,
or statsmodels; the downstream public biological validation module uses
pandas/scipy/requests for public-data analysis and API access.
This design provides:

- **Bit-exact reproducibility** across platforms and Python versions.
- **A minimal environment footprint** (NumPy + Matplotlib).
- **Full auditability** of every gradient and optimisation step.

---

## Requirements

| Package    | Version    | Purpose |
|------------|------------|---------|
| Python     | ≥ 3.10     | Runtime |
| NumPy      | ≥ 1.24, < 2.0 | Numerical backend |
| Matplotlib | ≥ 3.7      | Figure generation |

**Optional** (enable in `environment.yml` if needed):

| Package   | Version | Purpose |
|-----------|---------|---------|
| gseapy    | ≥ 1.0   | GSEA-style pathway enrichment (Step 6) |
| lifelines | ≥ 0.27  | Cox / log-rank survival analysis (Step 6) |
| pandas, scipy, requests, PyYAML | latest | Downstream public biological validation |
| GEOparse, pandas, requests | latest | Downloading GEO SOFT files (Step 8 `prepare-geo` only) |

---

## Installation

```bash
git clone https://github.com/<your-org>/CAGE-ESCA.git
cd CAGE-ESCA

# Create and activate the conda environment
conda env create -f environment.yml
conda activate cage

# Install in editable mode
pip install -e .

# Verify the installation
python -c "import cage; print(cage.__version__)"
```

---

## Obtaining Input Data

**No raw TCGA or GEO data are distributed with this repository.** Users must
download the source files from their official providers and place them in the
`data/` directory as described below. All file paths shown in the run commands
assume this layout.

Create the local data directory:

```bash
mkdir -p data/tcga data/geo data/msigdb data/ppi
```

### TCGA-ESCA RNA-seq (primary cohort)

Source: **NCI Genomic Data Commons (GDC) Data Portal** — project `TCGA-ESCA`.

- Portal: https://portal.gdc.cancer.gov/
- Programmatic access: `gdc-client` or the GDC REST API.

Three files are required:

| File | Content | Expected dimensions |
|------|---------|---------------------|
| `TCGA_ESCA_STAR_Counts.csv` | STAR gene-level raw counts | genes × 197 samples |
| `ESCA_vst_normalized_matrix.csv` | DESeq2 variance-stabilising transformation of the counts above | genes × 197 samples |
| `TCGA_ESCA_Metadata.csv` | Clinical, demographic, and molecular metadata | 197 samples × 137 covariates |

The VST matrix is produced by running DESeq2's `vst()` (blind = FALSE) on the
raw counts. A reference R script is provided in `scripts/prepare_tcga_vst.R`
(if applicable to your distribution) or can be generated by any standard
DESeq2 workflow.

**Placement** (paths used by the default commands):

```
data/tcga/
├── TCGA_ESCA_STAR_Counts.csv
├── ESCA_vst_normalized_matrix.csv
└── TCGA_ESCA_Metadata.csv
```

When running Step 2, pass `--input-dir data/tcga` (or override individual files
with `--counts-csv`, `--vst-csv`, `--metadata-csv`).

### GEO external validation cohorts

Source: **NCBI Gene Expression Omnibus (GEO)** — https://www.ncbi.nlm.nih.gov/geo/

Four ESCC cohorts are used for external replication:

| Accession | Platform | Tumour | Normal |
|-----------|----------|--------|--------|
| GSE38129  | Affymetrix HG-U133A 2.0 | 30 | 30 |
| GSE161533 | Affymetrix HG-U133 Plus 2.0 | 56 | 28 |
| GSE53624  | Agilent-038314 lncRNA + mRNA | 119 | 119 |
| GSE53625  | Agilent-038314 lncRNA + mRNA | 179 | 179 |

CAGE provides an automated downloader that fetches SOFT.gz archives, extracts
expression matrices, and infers tumour/normal labels:

```bash
python -m cage.step8_external_validation_and_release prepare-geo \
  --gse GSE38129 GSE161533 GSE53624 GSE53625 \
  --output-dir data/geo
```

Manual download alternative: fetch `<GSE>_family.soft.gz` from
`https://ftp.ncbi.nlm.nih.gov/geo/series/<GSEnnn>/<GSE>/soft/` and place each
file at `data/geo/<GSE>/raw/<GSE>_family.soft.gz` before running `prepare-geo`.

### Pathway databases (MSigDB)

Pathway gene-set files are **not** distributed with the repository. Download
them directly from the official **Molecular Signatures Database (MSigDB)**:

- Site: https://www.gsea-msigdb.org/gsea/msigdb/
- Register (free) and download the human (`Hs.symbols.gmt`) versions.

| Collection | File on MSigDB | Local path used by Step 6 |
|------------|----------------|--------------------------------|
| **Hallmark** | `h.all.<version>.Hs.symbols.gmt` | `data/msigdb/h.all.Hs.symbols.gmt` |
| **KEGG** (Medicus or Legacy) | `c2.cp.kegg_medicus.<version>.Hs.symbols.gmt` | `data/msigdb/c2.cp.kegg.Hs.symbols.gmt` |
| **Reactome** | `c2.cp.reactome.<version>.Hs.symbols.gmt` | `data/msigdb/c2.cp.reactome.Hs.symbols.gmt` |
| **ImmuneSigDB** | `c7.immunesigdb.<version>.Hs.symbols.gmt` | `data/msigdb/c7.immunesigdb.Hs.symbols.gmt` |

Pass the paths to Step 6 via `--gmt-hallmark`, `--gmt-kegg-go`,
`--gmt-reactome`, and `--gmt-immune` (see the Step 6 example below).
Pathway analysis is optional — omit the flags to skip enrichment.

### Optional: protein–protein interaction edge list

Network-level support (Step 6 `--run-network`) requires a tab-separated
edge list with at minimum two columns of gene symbols. Suitable sources include:

- **STRING** — https://string-db.org/ (download `protein.links.v*.txt`, then
  map Ensembl protein IDs to gene symbols and threshold by combined score).
- **BioGRID** — https://thebiogrid.org/ (download `BIOGRID-ALL-*.tab3.txt`).
- **HuRI / HumanNet** — https://www.interactome-atlas.org/, https://www.inetbio.org/humannet/

Save the prepared edge list as `data/ppi/ppi_edges.tsv` and pass it via
`--ppi-edge-list data/ppi/ppi_edges.tsv`.

---

## Repository Layout

```
CAGE-ESCA/
├── cage/                     # Pipeline package (all Step 2–8 modules)
├── tests/                    # Unit and CLI smoke tests
├── data/                     # (User-provided) input data — not version-controlled
├── outputs/                  # Generated outputs (created on first run)
├── environment.yml           # Conda environment specification
├── pyproject.toml            # Package metadata and entry points
├── requirements.txt          # pip dependency pin
└── README.md
```

The `data/` directory should be added to `.gitignore` to ensure that no raw
TCGA or GEO data are committed to version control.

---

## Pipeline Architecture

```
 Step 2 ─► Step 3 ─► Step 4 ─► Step 5 ─► Step 6 ─► (Step 6b)─► Step 7 ─► Step 8
 Cohort    Baselines  Deep      CDPS      Bio.       Top-25     Manuscript External
 curation             model     ranking   validation prioriti-  packaging  validation
                                                     sation                & release
```

### Model architecture (Step 4)

```
Input (filtered genes)
        │
   ┌────▼────┐
   │ Feature │   L1 or hard-concrete
   │  Gate   │   sparsity regulariser
   └────┬────┘
        │
   ┌────▼────┐
   │ Encoder │   hidden 256 → 96 → 48 (latent)
   └────┬────┘
        │
   ┌────┼────────────┬────────────────┐
   ▼    ▼            ▼                ▼
Classifier  Adversary   Invariance   Decoder
(tumour /   (confounder (latent-mean (optional
 normal)    via GRL)    penalty)     reconstruction)
```

---

## Quick Start

The commands below execute the full pipeline end-to-end. Adjust `--input-dir`
to point at the local `data/tcga` directory created above.

```bash
# Step 2 — Cohort curation and preprocessing
python -m cage.step2_build_cohort \
  --input-dir data/tcga \
  --output-dir outputs/step2_cohort \
  --n-top-variable-genes 5000 \
  --n-outer-folds 5 --n-inner-folds 3 \
  --run-cohort-flow-figure --seed 2026

# Step 3 — Grouped baseline classifiers
python -m cage.step3_grouped_baselines \
  --input-dir outputs/step2_cohort \
  --output-dir outputs/step3_baselines \
  --models logistic elasticnet decision_tree \
  --run-calibration --run-subgroup-sensitivity

# Step 4 — Sparse invariant deep model
python -m cage.step4_sparse_invariant_model \
  --input-dir outputs/step2_cohort \
  --output-dir outputs/step4_deep_model --overwrite \
  --latent-dim 32 --hidden-dims 128 64 \
  --n-epochs 100 --batch-size 64 --lr 8e-4 --weight-decay 5e-4 \
  --dropout 0.25 --patience 20 \
  --sparsity-lambda 0.01 --adv-lambda 0.05 \
  --invariance-lambda 0.001 --recon-lambda 0.15 --adv-ramp-epochs 15 \
  --run-calibration --run-subgroup-sensitivity

# Step 5 — CDPS ranking
python -m cage.step5_cdps_ranking \
  --step2-dir outputs/step2_cohort \
  --step4-dir outputs/step4_deep_model \
  --output-dir outputs/step5_cdps \
  --attribution-method integrated-gradients \
  --top-ks 25 --run-perturbation

# Step 6 — Biological validation (pathway files from MSigDB)
python -m cage.step6_biological_validation \
  --step2-dir outputs/step2_cohort \
  --step5-dir outputs/step5_cdps \
  --output-dir outputs/step6_validation \
  --run-enrichment --run-network --run-survival --run-subgroup-sensitivity \
  --gmt-hallmark data/msigdb/h.all.Hs.symbols.gmt \
  --gmt-reactome data/msigdb/c2.cp.reactome.Hs.symbols.gmt \
  --gmt-kegg-go  data/msigdb/c2.cp.kegg.Hs.symbols.gmt \
  --gmt-immune   data/msigdb/c7.immunesigdb.Hs.symbols.gmt \
  --ppi-edge-list data/ppi/ppi_edges.tsv

# Step 7 — Manuscript packaging
python -m cage.step7_manuscript_packaging \
  --step2-dir outputs/step2_cohort \
  --step3-dir outputs/step3_baselines \
  --step4-dir outputs/step4_deep_model \
  --step5-dir outputs/step5_cdps \
  --step6-dir outputs/step6_validation \
  --output-dir outputs/final_package \
  --export-markdown --copy-final-figures --copy-key-tables --build-supplement

# Step 8 — External validation and release
python -m cage.step8_external_validation_and_release prepare-geo \
  --gse GSE38129 GSE161533 GSE53624 GSE53625 \
  --output-dir data/geo

python -m cage.step8_external_validation_and_release validate-geo \
  --geo-dir data/geo \
  --step5-dir outputs/step5_cdps --step6-dir outputs/step6_validation \
  --step4-dir outputs/step4_deep_model --run-model \
  --output-dir outputs/step8_geo_validation

python -m cage.step8_external_validation_and_release validate-agilent \
  --geo-dir data/geo \
  --step5-dir outputs/step5_cdps --step6-dir outputs/step6_validation \
  --step4-dir outputs/step4_deep_model --run-model \
  --output-dir outputs/step8_agilent_validation

python -m cage.step8_external_validation_and_release release \
  --step5-dir outputs/step5_cdps --step6-dir outputs/step6_validation \
  --step7-dir outputs/final_package \
  --output-dir outputs/step8_release \
  --release-bundle-dir outputs/release_bundle \
  --copy-final-figures --copy-key-tables --build-supplement

# Step 9 - Public biological validation and composite figure
cage-public-validation \
  --genes FOXS1 ESM1 KIF2C \
  --hpa data/HPA/proteinatlas.tsv \
  --output outputs/step9_public_biological_validation

cage-public-validation-figure \
  --step9-dir outputs/step9_public_biological_validation \
  --output-dir outputs/step9_public_biological_validation/composite
```

Every module supports `--help`:

```bash
python -m cage.<module_name> --help
```

---

## Module Reference

Each entry summarises **Purpose**, **Inputs**, **Outputs**, and the most
useful flags. All modules accept the [Global CLI Options](#global-cli-options).

### `cage.step2_build_cohort` — Cohort curation and preprocessing

- **Purpose.** Assemble a leakage-free TCGA-ESCA cohort: harmonise barcodes,
  retain primary tumour (`01`) and solid-normal (`11`) samples, apply a
  variance filter, optionally cap genes by variance, encode confounders, and
  generate patient-grouped nested cross-validation folds.
- **Inputs.** Three CSVs in `--input-dir`: STAR raw counts, VST-normalised
  expression, and clinical metadata (see [Obtaining Input Data](#obtaining-input-data)).
- **Outputs.** `master_samples_primary.csv`, `normalized_primary_matrix.csv`,
  `counts_primary_matrix.csv`, `grouped_outer_folds.csv`,
  `confounder_encodings.json`, `phase1_summary.json`, and a cohort-flow figure.
- **Key flags.** `--n-top-variable-genes`, `--use-all-filtered-genes`,
  `--rigor-profile {standard,all_genes}`, `--n-outer-folds`, `--n-inner-folds`,
  `--environments`, `--run-cohort-flow-figure`.

### `cage.step2_gene_sensitivity` — Gene-cap sensitivity analysis (optional)

- **Purpose.** Quantify how the top-gene set varies when the variable-gene cap
  is changed; produces a Jaccard comparison across caps.
- **Inputs.** Same as Step 2.
- **Outputs.** `gene_cap_sensitivity_metrics.csv` and a Jaccard figure.
- **Usage.** `--gene-sensitivity-grid 5000 10000 20000 all`.

### `cage.step3_grouped_baselines` — Grouped baseline classifiers

- **Purpose.** Train patient-grouped baseline models as reference benchmarks
  (logistic ridge, elastic-net logistic, decision tree, optional random forest)
  with per-fold and overall out-of-fold metrics.
- **Inputs.** Step 2 output directory.
- **Outputs.** `baseline_oof_predictions.csv`, `baseline_per_fold_metrics.csv`,
  `baseline_summary_metrics.csv`, `baseline_feature_importance.csv`,
  optional calibration and subgroup tables, ROC/PR/calibration figures.
- **Key flags.** `--models`, `--decision-threshold`, `--bootstrap-ci-n`,
  `--run-calibration`, `--run-subgroup-sensitivity`.

### `cage.step4_sparse_invariant_model` — Sparse invariant deep model

- **Purpose.** Train the Sparse Invariant Adversarial Autoencoder-Classifier
  combining a per-gene gate, a tumour/normal classifier head, a confounder
  adversary (gradient reversal), an environment-invariance penalty, and an
  optional reconstruction decoder.
- **Inputs.** Step 2 output directory.
- **Outputs.** `deep_oof_predictions.csv`, `gate_weights.csv`,
  `latent_embeddings.csv`, per-fold metrics, training history, checkpoints,
  and latent / gate / ROC figures.
- **Key flags.** `--latent-dim`, `--hidden-dims`, `--sparsity-type {l1,hard-concrete}`,
  `--dropout`, `--n-epochs`, `--lr`, `--weight-decay`, `--patience`,
  `--sparsity-lambda`, `--adv-lambda`, `--invariance-lambda`, `--recon-lambda`,
  `--adv-ramp-epochs`, `--confounder-column`, `--environment-column`.

Composite loss:

```
L = L_clf + λ_s · ||g||_1 + λ_a · L_adv + λ_i · mean_k ||μ_k − μ||² + λ_r · L_recon
```

### `cage.step5_cdps_ranking` — Candidate Driver Priority Score

- **Purpose.** Integrate deep-model signals into a weighted composite ranking
  combining attribution, gate weight, cross-fold stability, environment
  invariance, and in-silico perturbation sensitivity.
- **Inputs.** Step 2 cohort and Step 4 trained model.
- **Outputs.** `ranked_genes_cdps.csv`, `top25_genes_cdps.csv`,
  `top100_genes_cdps.csv`, per-component score tables, and ranking figures.
- **Key flags.** `--attribution-method {integrated-gradients,grad-x-input,gate-weight}`,
  `--ig-steps`, `--stability-top-frac`, component weights (`--w-attribution`,
  `--w-gate`, `--w-stability`, `--w-invariance`, `--w-perturbation`),
  `--top-ks`, `--run-perturbation`.

Composite score:

```
CDPS = 0.30·attribution + 0.20·gate + 0.20·stability + 0.15·invariance + 0.15·perturbation
```

### `cage.step6_biological_validation` — Biological validation

- **Purpose.** Cross-validate top CDPS genes using independent biological
  evidence: differential expression, pathway enrichment (Hallmark, Reactome,
  KEGG, ImmuneSigDB), PPI network support, clinical association, optional
  survival analysis, and subgroup robustness.
- **Inputs.** Step 2 cohort, Step 5 ranking, MSigDB `.gmt` files, optional
  PPI edge list.
- **Outputs.** `differential_expression_results.csv`,
  `enrichment_results_<collection>.csv`, `network_gene_support.csv`,
  `clinical_association_results.csv`, `survival_gene_summary.csv`,
  `final_validated_gene_ranking.csv`, and validation figures.
- **Key flags.** `--de-method {welch,ranksum,nb}`, `--fdr-threshold`,
  `--lfc-threshold`, `--run-enrichment`, `--run-network`, `--run-survival`,
  `--run-subgroup-sensitivity`, `--gmt-hallmark`, `--gmt-reactome`,
  `--gmt-kegg-go`, `--gmt-immune`, `--ppi-edge-list`.

### `cage.step6b_top25_final_prioritization` — Top-25 evidence integration (optional)

- **Purpose.** Combine six independent evidence streams into a single
  evidence-weighted priority for the top-25 CDPS genes, with tiered
  classification (Tier 1 / 2 / 3) and a manuscript-ready table.
- **Inputs.** Step 5 CDPS, Step 6 validation, optional Step 8 GEO / Agilent
  replication outputs.
- **Outputs.** `top25_integrated_evidence.csv`,
  `top25_final_priority_ranking.csv`, `top25_tier_summary.csv`,
  `top25_manuscript_table.csv`, eight summary figures.
- **Key flags.** `--top-k`, `--tier1-threshold`, `--tier2-threshold`, and
  per-component weights (`--w-cdps`, `--w-external`, `--w-de`,
  `--w-clinical-survival`, `--w-pathway-network`, `--w-subgroup`).

### `cage.step6b_survival_external_validation` — Survival and external boxplots (optional)

- **Purpose.** Kaplan–Meier survival analysis on the TCGA cohort for the top-25
  prioritised genes, plus tumour-vs-normal boxplots in TCGA and in four
  independent GEO datasets. Produces individual figures and combined 5×5
  panels suitable for supplementary material.
- **Inputs.** TCGA normalised expression matrix and metadata; top-25 ranking
  from Step 6b; Step 8 replication tables; prepared GEO matrices.
- **Outputs.** `figures/km/`, `figures/boxplots/`, `figures/external/`,
  `top25_survival_summary.csv`, `top25_external_validation.csv`,
  `top25_full_validation_summary.csv`.

### `cage.step7_manuscript_packaging` — Manuscript assembly

- **Purpose.** Aggregate the outputs from Steps 2–6 into a publication-ready
  package: main figures, main tables, supplementary tables, Markdown drafts,
  and reproducibility manifests.
- **Inputs.** All upstream step output directories.
- **Outputs.** `final_figures/` (Figures 1–8), `final_tables/` (Tables 1–4 +
  S1–S6), `manuscript/manuscript_report.md`, `manuscript/manuscript_methods.md`,
  `manuscript/manuscript_results_summary.md`, `supplementary/`, `manifests/`.
- **Key flags.** `--export-markdown`, `--export-html`, `--export-docx-ready`,
  `--export-latex-ready`, `--copy-final-figures`, `--copy-key-tables`,
  `--build-supplement`, `--build-reviewer-bundle`.

### `cage.step8_external_validation_and_release` — External validation and release

A single CLI exposing four sub-commands:

| Sub-command | Purpose |
|-------------|---------|
| `prepare-geo` | Download GEO SOFT.gz files, extract expression matrices, infer tumour/normal labels, and map probes to gene symbols. |
| `validate-geo` | Affymetrix two-cohort (GSE38129, GSE161533) DE direction concordance and optional classifier transfer. |
| `validate-agilent` | Agilent two-cohort (GSE53624, GSE53625) validation; resolves probe-to-gene mapping via 60-mer matching against NCBI RefSeq. |
| `release` | Assemble the release bundle (figures, tables, manuscript, supplement, manifests). |
| `all` | Run all four sub-commands sequentially. |

Standalone helper modules (`step8_prepare_geo`, `step8_geo_validation`,
`step8_agilent_validation`) are also importable, but the unified CLI is the
recommended entry point.

### `cage.public_biological_validation` - Public biological validation

- **Purpose.** Run downstream public-data validation for final CAGE/CDPS
  candidates using HPA, cBioPortal TCGA-ESCA, immune/TME marker correlations,
  optional TISCH2 processed single-cell data, and an integrated validation
  summary.
- **Inputs.** Candidate genes, `data/HPA/proteinatlas.tsv`, TCGA-ESCA
  normalized expression matrix detected from common project locations, and
  optional TISCH2 processed files under `data/TISCH2/`.
- **Outputs.** `outputs/step9_public_biological_validation/` with separate
  `hpa/`, `cbioportal/`, `immune/`, `singlecell/`, `integrated/`, and
  `composite/` subdirectories.
- **CLI.** `cage-public-validation` runs the public-data validation tables and
  panels. `cage-public-validation-figure` assembles the composite manuscript
  figure from Step 9 outputs.

### Other modules

| Module | Role |
|--------|------|
| `cage.preprocess_esca` | Shared preprocessing helpers used by Step 2. |
| `cage.baseline_models` | Pure-NumPy baseline classifier implementations (Step 3). |
| `cage.deep_model_utils` | Layers, optimiser, and training utilities for Step 4. |
| `cage.metrics` | AUROC, AUPRC, balanced accuracy, Brier score, calibration, bootstrap CIs. |
| `cage.cli_args` | Shared argparse builders for cross-step CLI consistency. |
| `cage.publication_style` | Centralised Matplotlib styling (fonts, colours, sizes). |
| `cage.public_biological_validation` | Step 9 public-data biological validation CLI backend. |
| `cage.public_validation_composite` | Step 9 composite public-validation figure CLI backend. |
| `cage.step3_runner`, `step4_runner`, `step5_runner`, `step6_runner`, `step7_runner`, `step8_runner` | Convenience wrappers that orchestrate a single step end-to-end. |
| `cage.step4_ablation`, `step4_hptuner` | Optional ablation studies and hyperparameter sweeps for Step 4. |
| `cage.step5_robustness` | Robustness / perturbation diagnostics for the CDPS ranking. |

---

## Global CLI Options

All pipeline steps share a common set of options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | (required\*) | Directory containing inputs for this step. |
| `--output-dir` | (required) | Directory for outputs (tables, figures, logs). |
| `--seed` | `2026` | Random seed for reproducibility. |
| `--n-threads` | half CPU count | Worker threads; propagates to `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`. |
| `--overwrite` | off | Regenerate outputs even if they already exist. |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `--version` | — | Print the CAGE version and exit. |

\* Steps 5–8 accept per-phase `--stepN-dir` flags instead of `--input-dir`.

---

## Figure and Visualization Standards

CAGE writes all figures through a single style module
(`cage/publication_style.py`) so that visual settings remain consistent across
the pipeline and can be tuned from a single place. Defaults are conservative
and journal-agnostic:

| Property | Default |
|----------|---------|
| Primary format | SVG |
| Additional formats | PDF, PNG (via `--extra-figure-format`) |
| Font family | DejaVu Sans (falls back gracefully if a system font is unavailable) |
| Base font size | 10 pt |
| Colour palette | CAGE default categorical palette (colour-blind-friendly where feasible) |
| Design | Minimal axes, no chart junk, transparent backgrounds |
| Raster DPI | 300 |

All settings are configurable from the command line:

| Flag | Default | Description |
|------|---------|-------------|
| `--figure-format` | `svg` | Primary output format. |
| `--extra-figure-format` | — | Repeatable additional format(s), e.g. `--extra-figure-format pdf --extra-figure-format png`. |
| `--font-family` | DejaVu Sans | Any font installed on the system. |
| `--font-size` | `10` | Base font size in points. |
| `--palette` | `cage` | Categorical palette name (`cage` default; `nature` retained as a backward-compatible alias). |

Users targeting a specific journal can override fonts without modifying the
source — for example `--font-family "Times New Roman" --font-size 9`.

---

## Output Directory Structure

```
outputs/
├── step2_cohort/                       # cohort, folds, encodings, preprocessing report
├── step3_baselines/                    # baseline OOF predictions, metrics, importances
├── step4_deep_model/                   # deep-model predictions, gate weights, latents, checkpoints
├── step5_cdps/                         # CDPS ranking and component scores
├── step6_validation/                   # DE, enrichment, network, clinical, survival
├── step6b_top25_prioritization/        # optional — integrated top-25 priority ranking
├── step6b_survival_external/           # optional — KM curves and external boxplots
├── final_package/                      # manuscript package (figures, tables, drafts)
├── step8_geo_validation/               # Affymetrix replication and classifier transfer
├── step8_agilent_validation/           # Agilent replication and classifier transfer
├── step8_release/                      # release bundle assembly
└── release_bundle/                     # final shareable directory (figures, tables, manuscript, manifests)
```

Step 9 public biological validation writes to
`outputs/step9_public_biological_validation/`, with separate `hpa/`,
`cbioportal/`, `immune/`, `singlecell/`, `integrated/`, and `composite/`
subdirectories.

Each step also writes a `logs/` subdirectory and a `phaseN_summary.json`
machine-readable summary of the configuration and key results.

---

## Reproducibility

CAGE is designed for **bit-exact, end-to-end reproducibility**:

- **Deterministic seeds.** All random operations (fold assignment, weight
  initialisation, bootstrap sampling, perturbation) are seeded via `--seed`
  (default `2026`).
- **Patient-grouped CV.** Outer folds are grouped by patient barcode to
  prevent leakage between train and test splits.
- **Thread-limit propagation.** `--n-threads` is set before NumPy import so
  BLAS-level non-determinism is suppressed.
- **Phase summaries.** Each step writes `phaseN_summary.json` capturing
  parameters, sample counts, metrics, and output artefacts.
- **Manifests.** Step 7 produces `output_manifest.csv`, `figure_manifest.csv`,
  and `table_manifest.csv` enumerating every generated artefact.
- **Logging.** Every step logs its configuration and thresholds both to the
  console and to a durable `logs/` file.
- **Pure-NumPy backend.** No GPU non-determinism, no framework-version drift.

---

## Testing

A small test suite covers metric implementations, fold construction, leakage
checks, CDPS scoring, and CLI smoke tests:

```bash
pytest -q
```
---

## Citation

If you use CAGE in your research, please cite:

> [Muahmmad Waqas et al]. CAGE: Causality-Aware Gene Evaluator — prioritising candidate
> driver genes in esophageal carcinoma via sparse invariant deep modelling and
> multi-layer biological validation. *Bioinformatics* (2026). [DOI pending]

---

## License

This project is released under the [MIT License](LICENSE).
