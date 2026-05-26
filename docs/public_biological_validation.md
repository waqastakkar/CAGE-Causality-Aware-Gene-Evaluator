# Public Biological Validation Module

## Purpose

This downstream module adds reproducible public-data validation for final CAGE/CDPS ESCA candidate genes after prioritization. It is designed for manuscript strengthening when wet-lab validation is not available. It keeps the core CAGE model training and CDPS workflow separate.

Default candidate genes:

- FOXS1
- ESM1
- KIF2C

## Required Input Files

- `data/HPA/proteinatlas.tsv`
- A processed TCGA ESCA expression matrix for immune/TME marker correlation analysis

The expression matrix is detected automatically from common locations, including:

- `data/processed/`
- `results/`
- `outputs/`
- repository root

Likely file names include `ESCA_vst_normalized_matrix.csv`, `vst_normalized`, `expression_matrix`, `normalized_expression`, `TCGA_ESCA`, or `normalized_primary_matrix`.

## Optional Input Files

Processed single-cell or TISCH2 data can be placed in:

- `data/TISCH2/`
- `data/external/TISCH2/`

Supported file names may include:

- `expression_matrix.*`
- `cell_metadata.*`
- `cell_type_annotation.*`
- `cluster_average_expression.*`
- `metadata.*`

If no processed single-cell data are found, the module writes `outputs/step9_public_biological_validation/singlecell/singlecell_validation_skipped.txt` and continues.

## How To Run

From the repository root:

```bash
cage-public-validation \
  --genes FOXS1 ESM1 KIF2C \
  --hpa data/HPA/proteinatlas.tsv \
  --output outputs/step9_public_biological_validation
```

Equivalent module form:

```bash
python -m cage.public_biological_validation \
  --genes FOXS1 ESM1 KIF2C \
  --hpa data/HPA/proteinatlas.tsv \
  --output outputs/step9_public_biological_validation
```

Create the composite manuscript figure from Step 9 outputs:

```bash
cage-public-validation-figure \
  --step9-dir outputs/step9_public_biological_validation \
  --output-dir outputs/step9_public_biological_validation/composite
```

The cBioPortal step uses the public API with study ID `esca_tcga`. If a cBioPortal endpoint fails, remaining endpoints continue and the error is written to `outputs/step9_public_biological_validation/cbioportal/cbioportal_api_log.txt`.

## Expected Outputs

```text
outputs/step9_public_biological_validation/
├── public_biological_validation.log
├── manuscript_figure_data_manifest.csv
├── hpa/
│   ├── hpa_candidate_gene_summary.csv
│   ├── hpa_available_columns.csv
│   ├── hpa_candidate_evidence_plot_data.csv
│   ├── hpa_missing_fields_report.txt
│   ├── figure_hpa_candidate_evidence_heatmap.svg
│   └── figure_hpa_candidate_evidence_heatmap.png
├── cbioportal/
│   ├── cbioportal_molecular_profiles.csv
│   ├── cbioportal_mutations.csv
│   ├── cbioportal_cna.csv
│   ├── cbioportal_mrna.csv
│   ├── cbioportal_clinical.csv
│   ├── cbioportal_candidate_gene_summary.csv
│   ├── cbioportal_alteration_frequency_plot_data.csv
│   ├── cbioportal_api_log.txt
│   ├── figure_cbioportal_candidate_gene_alteration_frequency.svg
│   ├── figure_cbioportal_candidate_gene_alteration_frequency.png
│   ├── figure_cbioportal_candidate_gene_evidence_heatmap.svg
│   └── figure_cbioportal_candidate_gene_evidence_heatmap.png
├── immune/
│   ├── immune_marker_gene_correlations.csv
│   ├── immune_marker_gene_correlations_fdr.csv
│   ├── immune_validation_summary.csv
│   ├── figure_candidate_gene_immune_correlation_heatmap.svg
│   ├── figure_candidate_gene_immune_correlation_heatmap.png
│   └── immune_validation_log.txt
├── singlecell/
│   ├── singlecell_gene_by_celltype.csv
│   ├── singlecell_percent_expressing.csv
│   ├── singlecell_candidate_gene_summary.csv
│   ├── figure_singlecell_dotplot.svg
│   ├── figure_singlecell_heatmap.svg
│   └── singlecell_validation_skipped.txt
└── integrated/
    ├── candidate_gene_public_validation_summary.csv
    ├── candidate_gene_biological_interpretation.md
    ├── figure_integrated_public_validation_heatmap.svg
    └── figure_integrated_public_validation_heatmap.png
```

Single-cell CSV and figure outputs are produced only when compatible processed data are present.

## Validation Steps

1. HPA protein/RNA validation reads `proteinatlas.tsv` directly, detects relevant columns automatically, saves all available HPA fields for FOXS1, ESM1, and KIF2C, and logs missing evidence groups without failing.
2. cBioPortal validation retrieves molecular profiles, mutations, copy-number alterations, mRNA data, and clinical/sample data from the API when available.
3. Immune/TME validation calculates Pearson and Spearman correlations between candidate genes and immune, stromal, endothelial, checkpoint, angiogenesis, epithelial, and proliferation markers, with Benjamini-Hochberg FDR correction.
4. Optional single-cell validation summarizes candidate expression by cell type if processed TISCH2 or compatible single-cell files are present.
5. Integrated summary combines HPA, cBioPortal, immune/TME, and optional single-cell evidence into tables and figures.
