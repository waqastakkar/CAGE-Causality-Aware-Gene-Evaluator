"""CAGE: Causality-Aware Gene Evaluator.

A reproducible, end-to-end computational pipeline for prioritizing candidate
driver genes in TCGA esophageal carcinoma (ESCA). CAGE integrates cohort
curation, patient-grouped cross-validation, sparse invariant deep modelling,
multi-component Causality-Aware Priority Score (CDPS) ranking, biological
validation, and orthogonal external cohort replication — all in pure NumPy.

Pipeline steps (accessible as ``python -m cage.<step_module>``):
    step2_build_cohort
    step3_grouped_baselines
    step4_sparse_invariant_model
    step5_cdps_ranking
    step6_biological_validation
    step7_manuscript_packaging
    step8_external_validation_and_release  (unified subcommand CLI)
      └─ step8_prepare_geo        (prepare-geo subcommand)
      └─ step8_geo_validation     (validate-geo subcommand)
      └─ step8_agilent_validation (validate-agilent subcommand)
      └─ step8_runner             (release subcommand backend)
    step6b_top25_final_prioritization      (post-analysis, optional)
    step6b_survival_external_validation    (post-analysis, optional)
"""

__version__ = "0.1.0"
__all__ = [
    "step2_build_cohort",
    "step3_grouped_baselines",
    "step3_runner",
    "step4_sparse_invariant_model",
    "step4_runner",
    "step5_cdps_ranking",
    "step5_runner",
    "step6_biological_validation",
    "step6_runner",
    "step6b_top25_final_prioritization",
    "step6b_survival_external_validation",
    "step7_manuscript_packaging",
    "step7_runner",
    "step8_external_validation_and_release",
    "step8_runner",
    "step8_prepare_geo",
    "step8_geo_validation",
    "step8_agilent_validation",
    "publication_style",
    "cli_args",
    "preprocess_esca",
    "metrics",
    "baseline_models",
    "deep_model_utils",
]
