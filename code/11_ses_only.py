"""11 — Sub-analysis: how well do socioeconomic/demographic factors ALONE predict spending (no health conditions)?
Nested feature sets on the same 80/20 split:  A = demographics + SES;  B = A + access (insurance, usual source of care);
C = B + health status & chronic conditions (full model).  Models: OLS, KNN (K=60, distance), XGBoost (CV-tuned params).
Outcomes: log total spending, log OOP spending (test R²), high-cost top-10% (test AUC). SHAP beeswarm for XGBoost on set A."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
df = pd.read_csv("data/meps2022_adults.csv")
SES_N = ["age", "female", "educ_yrs", "fam_income", "fam_size", "married", "employed"]; SES_C = ["race", "pov_cat", "region"]
ACC_N = ["has_usual_source"]; ACC_C = ["insurance"]
HLT_N = ["health_poor", "mental_poor", "diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer", "n_chronic"]
SETS = {"Demographics + SES": (SES_N, SES_C), "+ Access to care": (SES_N + ACC_N, SES_C + ACC_C),
        "+ Health conditions (full)": (SES_N + ACC_N + HLT_N, SES_C + ACC_C)}
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
w = df.weight.values; wt = w[te]
o = df.total_exp.sort_values(); cut = o[(df.weight[o.index].cumsum() / df.weight.sum()) >= .9].iloc[0]; yhc = (df.total_exp >= cut).astype(int).values
P = pd.read_csv("output/tables/s_xgb_best_params.csv").set_index("param").value.to_dict()
P = {k: (int(v) if k in ("n_estimators", "max_depth") else v) for k, v in P.items()}
rows, keep = [], {}
for sname, (num, cat) in SETS.items():
    X = pd.get_dummies(df[num + cat], columns=cat, dtype=float); X.columns = [c.replace("<", "lt") for c in X.columns]; X = X.fillna(X.iloc[tr].median()); Xtr, Xte = X.iloc[tr], X.iloc[te]
    for y_name in ["log_total_exp", "log_oop_exp"]:
        y = df[y_name].values
        for mname, mdl, kw in [("OLS", make_pipeline(StandardScaler(), LinearRegression()), "linearregression__sample_weight"),
                               ("KNN", make_pipeline(StandardScaler(), KNeighborsRegressor(60, weights="distance")), None),
                               ("XGBoost", XGBRegressor(random_state=42, n_jobs=4, tree_method="hist", **P), "sample_weight")]:
            fit_kw = {kw: w[tr]} if kw else {}
            mdl.fit(Xtr, y[tr], **fit_kw); r2 = r2_score(y[te], mdl.predict(Xte), sample_weight=wt)
            rows.append(dict(features=sname, outcome=y_name, model=mname, metric="R2", value=r2))
            if sname == "Demographics + SES" and y_name == "log_total_exp" and mname == "XGBoost": keep = dict(m=mdl, X=Xte)
    for mname, mdl, kw in [("Logit", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), "logisticregression__sample_weight"),
                           ("XGBoost", XGBClassifier(random_state=42, n_jobs=4, tree_method="hist", **P), "sample_weight")]:
        mdl.fit(Xtr, yhc[tr], **{kw: w[tr]}); auc = roc_auc_score(yhc[te], mdl.predict_proba(Xte)[:, 1], sample_weight=wt)
        rows.append(dict(features=sname, outcome="high_cost", model=mname, metric="AUC", value=auc))
R = pd.DataFrame(rows); R.round(4).to_csv("output/tables/ses_only_subanalysis.csv", index=False)
print(R.pivot_table(index=["outcome", "model"], columns="features", values="value", sort=False).round(3).to_string())

# ---------------- figure: (left) R² ladder; (right) SHAP beeswarm for SES-only XGBoost
sv = shap.TreeExplainer(keep["m"]).shap_values(keep["X"]); Xs = keep["X"]
NICE = {"age": "Age", "female": "Female", "educ_yrs": "Education (years)", "fam_income": "Family income", "fam_size": "Family size",
        "married": "Married", "employed": "Employed", "race_NH White": "Race: NH White", "race_Hispanic": "Race: Hispanic",
        "race_NH Black": "Race: NH Black", "race_NH Asian": "Race: NH Asian", "race_NH Other/Multi": "Race: NH other/multi",
        "pov_cat_High income": "Income: high", "pov_cat_Poor (lt100% FPL)": "Income: poor", "pov_cat_Middle income": "Income: middle",
        "pov_cat_Low income": "Income: low", "pov_cat_Near poor": "Income: near poor", "region_South": "Region: South",
        "region_West": "Region: West", "region_Northeast": "Region: Northeast", "region_Midwest": "Region: Midwest"}
fig = plt.figure(figsize=(15, 6.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=.45, left=.06, right=.97, top=.93, bottom=.1)
a = fig.add_subplot(gs[0]); sets = list(SETS); C = [style.TEAL, style.GOLD, style.RUST]
outs = [("log_total_exp", "R2", "Log total spending (R²)"), ("log_oop_exp", "R2", "Log OOP spending (R²)"), ("high_cost", "AUC", "High-cost top 10% (AUC)")]
bw = .26
for k, sname in enumerate(sets):
    vals = [R[(R.features == sname) & (R.outcome == o_) & (R.model == "XGBoost")].value.iloc[0] for o_, _, _ in outs]
    x = np.arange(3) + (k - 1) * bw; a.bar(x, vals, bw, color=C[k], label=sname)
    for xi, v in zip(x, vals): a.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=8)
a.set_xticks(range(3), [l for _, _, l in outs], fontsize=9); a.set_ylim(0, 1.0); a.set_ylabel("Test-set score (XGBoost)")
a.set_title("How much do socioeconomic factors alone explain?"); a.grid(axis="x", visible=False)
a.legend(loc="upper left", frameon=False, title="Predictors used", title_fontsize=9)
b = fig.add_subplot(gs[1]); imp = np.abs(sv).mean(0); order = np.argsort(imp)[::-1][:12]; cmap = plt.get_cmap("coolwarm"); rng = np.random.default_rng(1)
for row, j in enumerate(order[::-1]):
    s = sv[:, j]; v = Xs.iloc[:, j].values.astype(float); lo, hi = np.nanpercentile(v, [5, 95]); vn = np.clip((v - lo) / (hi - lo if hi > lo else 1), 0, 1)
    bins = np.round(s / .04).astype(int); cnt = pd.Series(bins).map(pd.Series(bins).value_counts()).values
    jit = (rng.random(len(s)) - .5) * np.minimum(.8, .06 * np.sqrt(cnt)); idx = rng.permutation(len(s))
    b.scatter(s[idx], row + jit[idx], c=vn[idx], cmap=cmap, s=4, alpha=.75, linewidths=0, rasterized=True)
b.axvline(0, color="#555", lw=.8); b.set_yticks(range(len(order)), [NICE.get(Xs.columns[j], Xs.columns[j]) for j in order[::-1]])
b.grid(axis="y", visible=False); b.grid(axis="x", visible=True); b.set_ylim(-.7, len(order) - .3)
b.set_xlabel("SHAP value (log points of total spending)"); b.set_title("SES-only model: what drives predictions")
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=b, pad=.02, fraction=.035, ticks=[0, 1]); cb.ax.set_yticklabels(["Low", "High"])
cb.set_label("Feature value", fontsize=9); cb.outline.set_visible(False)
style.save(fig, "ses_only_subanalysis", "Source: MEPS HC-243 (2022), adults 18+. Same 80/20 split and CV-tuned XGBoost as main models; test metrics survey-weighted. "
           "SES = income, poverty category, education, employment; access = insurance + usual source of care.")
