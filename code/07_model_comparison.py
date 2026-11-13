"""07 — Main Figure 6: test-set comparison of all prediction models for log total spending (incl. CV-tuned KNN)."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import pandas as pd, numpy as np, matplotlib.pyplot as plt
s = pd.read_csv("output/tables/s_regression_metrics.csv")[["model", "MSE", "R2"]]
k = pd.read_csv("output/tables/k_knn_metrics.csv"); k = k[k.target == "log_total_exp"]
kn = k[k.model.str.startswith("KNN")].iloc[0]; nul = k[k.model.str.startswith("Null")].iloc[0]
s = pd.concat([s, pd.DataFrame([{"model": "KNN", "MSE": kn.test_MSE, "R2": kn.test_R2}])]).sort_values("R2")
FAM = {"OLS": "Linear", "LASSO": "Regularized", "Ridge": "Regularized", "ElasticNet": "Regularized", "KNN": "Nearest neighbours",
       "RandomForest": "Tree ensemble", "GradBoost": "Tree ensemble", "XGBoost": "Tree ensemble", "MLP": "Neural net"}
COL = {"Linear": style.GRAY, "Regularized": style.NAVY, "Nearest neighbours": style.TEAL, "Tree ensemble": style.RUST, "Neural net": style.PLUM}
LAB = {"ElasticNet": "Elastic Net", "RandomForest": "Random forest", "GradBoost": "Gradient boosting", "MLP": "Neural net (MLP)"}
fig, ax = plt.subplots(figsize=(8, 4.8)); ax = [ax]
y = np.arange(len(s)); c = [COL[FAM[m]] for m in s.model]
ax[0].barh(y, s.R2, color=c, height=.62); ax[0].set_yticks(y, [LAB.get(m, m) for m in s.model])
for yi, v in zip(y, s.R2): ax[0].text(v + .004, yi, f"{v:.3f}", va="center", fontsize=8.5)
ax[0].set_xlim(0.30, 0.41); ax[0].set_xlabel("Test R² on log(1 + total spending)"); ax[0].set_title("Out-of-sample fit by model")
ax[0].grid(axis="y", visible=False); ax[0].grid(axis="x", visible=True)
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color=v, label=k_) for k_, v in COL.items()], loc="lower right", frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=1, title="Model family", title_fontsize=9)
fig.tight_layout()
style.save(fig, "s_model_comparison", f"Source: MEPS HC-243 (2022), adults 18+. 80/20 split; hyperparameters by 5-fold CV; test metrics survey-weighted. Null-model MSE = {nul.test_MSE:.2f}.")
print(s)
