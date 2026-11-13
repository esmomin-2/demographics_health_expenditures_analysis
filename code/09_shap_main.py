"""09 — Main Figure 8: SHAP beeswarm (dots) for XGBoost on log total spending + group contributions.
Refits XGBoost with the CV-tuned hyperparameters from 04_supervised (same features, split, weights), TreeSHAP on the test set."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/meps2022_adults.csv")
NUM = ["age", "female", "educ_yrs", "fam_income", "fam_size", "health_poor", "mental_poor", "married", "employed", "has_usual_source",
       "diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer", "n_chronic"]
CAT = ["race", "pov_cat", "insurance", "region"]
X = pd.get_dummies(df[NUM + CAT], columns=CAT, dtype=float)
X.columns = [c.replace('<', 'lt').replace('[', '(').replace(']', ')') for c in X.columns]
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
X = X.fillna(X.iloc[tr].median()); y = df.log_total_exp.values; w = df.weight.values
P = pd.read_csv("output/tables/s_xgb_best_params.csv").set_index("param").value.to_dict()
P = {k: (int(v) if k in ("n_estimators", "max_depth") else v) for k, v in P.items()}
m = XGBRegressor(random_state=42, n_jobs=4, tree_method="hist", **P).fit(X.iloc[tr], y[tr], sample_weight=w[tr])
Xt = X.iloc[te]; sv = shap.TreeExplainer(m).shap_values(Xt)
NICE = {"n_chronic": "No. chronic conditions", "has_usual_source": "Usual source of care", "fam_size": "Family size", "educ_yrs": "Education (years)",
        "female": "Female", "health_poor": "Poor self-rated health (1–5)", "age": "Age", "married": "Married", "mental_poor": "Poor mental health (1–5)",
        "diabetes": "Diabetes", "arthritis": "Arthritis", "employed": "Employed", "fam_income": "Family income", "hypertension": "Hypertension",
        "race_NH White": "Race: NH White", "race_Hispanic": "Race: Hispanic", "race_NH Black": "Race: NH Black", "race_NH Asian": "Race: NH Asian",
        "insurance_Uninsured": "Insurance: uninsured", "insurance_Any private": "Insurance: any private", "insurance_Public only": "Insurance: public only",
        "pov_cat_High income": "Income: high (≥400% FPL)", "region_South": "Region: South", "cancer": "Cancer", "asthma": "Asthma", "heart_disease": "Heart disease"}
imp = np.abs(sv).mean(0); order = np.argsort(imp)[::-1][:15]
fig, ax = plt.subplots(figsize=(10, 7.6))
rng = np.random.default_rng(42); cmap = plt.get_cmap("coolwarm")
for row, j in enumerate(order[::-1]):
    s = sv[:, j]; v = Xt.iloc[:, j].values.astype(float)
    lo, hi = np.nanpercentile(v, [5, 95]); vn = np.clip((v - lo) / (hi - lo if hi > lo else 1), 0, 1)
    # beeswarm-style vertical jitter proportional to local density
    bins = np.round(s / 0.04).astype(int); cnt = pd.Series(bins).map(pd.Series(bins).value_counts()).values
    jit = (rng.random(len(s)) - .5) * np.minimum(0.8, 0.06 * np.sqrt(cnt))
    idx = rng.permutation(len(s))
    ax.scatter(s[idx], row + jit[idx], c=vn[idx], cmap=cmap, s=5, alpha=.75, linewidths=0, rasterized=True)
ax.axvline(0, color="#555", lw=.8)
ax.set_yticks(range(len(order)), [NICE.get(X.columns[j], X.columns[j]) for j in order[::-1]])
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True); ax.set_ylim(-.7, len(order) - .3)
ax.set_xlabel("SHAP value (log points of total spending; + = predicted to spend more)")
ax.set_title("How each feature moves predicted spending (each dot = one adult)")
sm = plt.cm.ScalarMappable(cmap=cmap); cb = fig.colorbar(sm, ax=ax, pad=.02, fraction=.03, ticks=[0, 1])
cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Feature value", fontsize=9); cb.outline.set_visible(False)
fig.tight_layout()
style.save(fig, "s_shap_beeswarm_main", f"Source: MEPS HC-243 (2022), adults 18+. XGBoost (CV-tuned) on log(1 + total $); TreeSHAP on {len(te):,} test adults; top 15 of {X.shape[1]} features by mean |SHAP|. 0.1 log points ≈ 10.5%.")
print([(X.columns[j], round(imp[j], 3)) for j in order])
