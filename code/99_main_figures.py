"""99 — Collect the 10 main figures into output/main_figures (numbered); everything else -> output/figures/supplementary."""
import os, shutil, glob
F = "output/figures"; M = "output/main_figures"; S = f"{F}/supplementary"
os.makedirs(M, exist_ok=True); os.makedirs(S, exist_ok=True)
MAIN = [("f0_analysis_schematic", "figure_1_analysis_schematic"),
        ("f1_means_by_group", "figure_2_health_adjusted_spending_by_group"),
        ("drivers_domains", "figure_3_drivers_by_domain"),
        ("drivers_demographics_ses", "figure_4_demographic_ses_access_drivers"),
        ("o_race_income_insurance", "figure_5_oop_race_by_income"),
        ("o_nested_race_gaps", "figure_6_oop_race_gaps_nested_models"),
        ("interactions_race", "figure_7_race_gap_interactions"),
        ("u_embeddings_compare", "figure_8_profiles_pca_tsne_umap"),
        ("u_cluster_spending_adj", "figure_9_spending_by_profile"),
        ("pred_ses_vs_full", "figure_10_prediction_ses_vs_full"),
        ("k_knn_cv", "figure_11_knn_full_model"),
        ("knn_ses_only", "figure_12_knn_demographics_ses_only"),
        ("s_shap_beeswarm_main", "figure_13_shap_beeswarm"),
        ("i_dml_forest", "figure_14_dml_vs_ols"),
        ("f9_key_findings", "figure_15_key_findings_vs_literature")]
for f in glob.glob(f"{M}/*.png"): os.remove(f)
main_src = {a for a, _ in MAIN}
for a, b in MAIN:
    p = f"{F}/{a}.png" if os.path.exists(f"{F}/{a}.png") else f"{S}/{a}.png"
    print(("OK      " if os.path.exists(p) else "MISSING ") + b)
    if os.path.exists(p): shutil.copy(p, f"{M}/{b}.png")
for p in glob.glob(f"{F}/*.png"):  # move non-main to supplementary
    if True:
        if os.path.basename(p)[:-4] not in main_src: shutil.move(p, f"{S}/{os.path.basename(p)}")
