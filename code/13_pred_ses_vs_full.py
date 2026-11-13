"""13 — Prediction with DEMOGRAPHICS + SES only vs FULL feature set (adds access + health), same split for every model.
Models: OLS, LASSO, Ridge, Elastic Net (CV), KNN (CV over K), Random forest, Gradient boosting, XGBoost (tuned params), MLP.
Also: KNN on demographics + SES only -> CV curve, high-cost ROC, permutation-SHAP drivers.
Outputs: pred_ses_vs_full.png (model comparison), knn_ses_only.png; tables pred_ses_vs_full.csv, knn_ses_shap.csv."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, roc_auc_score, roc_curve
from xgboost import XGBRegressor
d = pd.read_csv("data/meps2022_adults.csv")
SES_N = ["age", "female", "educ_yrs", "fam_income", "fam_size", "married", "employed"]; SES_C = ["race", "pov_cat", "region"]
FULL_N = SES_N + ["has_usual_source", "health_poor", "mental_poor", "diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer", "n_chronic"]
FULL_C = SES_C + ["insurance"]
SETS = {"Demographics + SES only": (SES_N, SES_C), "Full (+ access + health)": (FULL_N, FULL_C)}
tr, te = train_test_split(d.index, test_size=0.2, random_state=42); w = d.weight; kf = KFold(5, shuffle=True, random_state=42)
y = d.log_total_exp
P = pd.read_csv("output/tables/s_xgb_best_params.csv").set_index("param").value.to_dict()
P = {k: (int(v) if k in ("n_estimators", "max_depth") else v) for k, v in P.items()}
def pre(num, cat): return ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                                             ("c", Pipeline([("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), ("s", StandardScaler())]), cat)])
MODELS = {"OLS": lambda: LinearRegression(), "LASSO": lambda: LassoCV(cv=kf, n_alphas=30), "Ridge": lambda: RidgeCV(alphas=np.logspace(-2, 4, 30)),
          "Elastic Net": lambda: ElasticNetCV(cv=kf, l1_ratio=[.2, .5, .8], n_alphas=20),
          "KNN": lambda: GridSearchCV(KNeighborsRegressor(), {"n_neighbors": [10, 25, 40, 60, 100, 150], "weights": ["uniform", "distance"]}, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1),
          "Random forest": lambda: RandomForestRegressor(300, max_depth=10, min_samples_leaf=20, max_features=.5, n_jobs=-1, random_state=42),
          "Gradient boosting": lambda: GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=.03, subsample=.8, random_state=42),
          "XGBoost": lambda: XGBRegressor(random_state=42, n_jobs=4, tree_method="hist", **P),
          "Neural net (MLP)": lambda: MLPRegressor((64, 32), alpha=1e-2, max_iter=400, early_stopping=True, random_state=42)}
rows, knn_store = [], {}
for sname, (num, cat) in SETS.items():
    for mname, mk in MODELS.items():
        pipe = Pipeline([("p", pre(num, cat)), ("m", mk())])
        fit_kw = {} if mname in ("KNN", "Neural net (MLP)", "LASSO", "Elastic Net", "Ridge") else {"m__sample_weight": w[tr].values}
        pipe.fit(d.loc[tr, num + cat], y[tr], **fit_kw)
        r2 = r2_score(y[te], pipe.predict(d.loc[te, num + cat]), sample_weight=w[te])
        rows.append(dict(features=sname, model=mname, test_R2=r2)); print(sname, mname, round(r2, 3), flush=True)
        if mname == "KNN": knn_store[sname] = pipe
R = pd.DataFrame(rows); R.round(4).to_csv("output/tables/pred_ses_vs_full.csv", index=False)

# ---------- Figure: model comparison, both feature sets
piv = R.pivot(index="model", columns="features", values="test_R2").sort_values("Full (+ access + health)")
fig, ax = plt.subplots(figsize=(10, 5.6)); yy = np.arange(len(piv)); h = .38
ax.barh(yy + h / 2, piv["Full (+ access + health)"], h, color=style.NAVY, label="Full (+ access + health)")
ax.barh(yy - h / 2, piv["Demographics + SES only"], h, color=style.TEAL, label="Demographics + SES only")
for i, (a, b) in enumerate(zip(piv["Full (+ access + health)"], piv["Demographics + SES only"])):
    ax.text(a + .004, i + h / 2, f"{a:.2f}", va="center", fontsize=8); ax.text(b + .004, i - h / 2, f"{b:.2f}", va="center", fontsize=8)
ax.set_yticks(yy, piv.index); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True); ax.set_xlim(0, .46)
ax.set_xlabel("Test R² on log(1 + total spending)"); ax.set_title("Out-of-sample fit: demographics + SES alone vs full model")
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DDD", framealpha=1, title="Predictors", title_fontsize=9)
fig.tight_layout(); style.save(fig, "pred_ses_vs_full", "Source: MEPS HC-243 (2022), adults 18+. Same 80/20 split for all models; penalties and K chosen by 5-fold CV; XGBoost uses CV-tuned settings; test R² survey-weighted. "
                                 "SES = income, poverty category, education, employment; demographics = age, sex, race, family size, marital status, region.")

# ---------- KNN on demographics + SES only: CV curve, high-cost ROC, SHAP drivers
num, cat = SETS["Demographics + SES only"]; X = d[num + cat]
cvr = pd.DataFrame(knn_store["Demographics + SES only"].named_steps["m"].cv_results_)
o = d.total_exp.sort_values(); cut = o[(w[o.index].cumsum() / w.sum()) >= .9].iloc[0]; yc = (d.total_exp >= cut).astype(int)
gc = GridSearchCV(Pipeline([("p", pre(num, cat)), ("m", KNeighborsClassifier())]), {"m__n_neighbors": [50, 100, 200, 300, 400], "m__weights": ["uniform"]},
                  cv=kf, scoring="roc_auc", n_jobs=-1).fit(X.loc[tr], yc[tr])
pk = gc.predict_proba(X.loc[te])[:, 1]; auc_k = roc_auc_score(yc[te], pk, sample_weight=w[te])
lg = Pipeline([("p", pre(num, cat)), ("m", LogisticRegression(max_iter=3000))]).fit(X.loc[tr], yc[tr]); pl = lg.predict_proba(X.loc[te])[:, 1]
auc_l = roc_auc_score(yc[te], pl, sample_weight=w[te]); print("SES-only high-cost AUC: KNN", round(auc_k, 3), gc.best_params_, "logit", round(auc_l, 3))
CODES = {c: sorted(d[c].dropna().unique()) for c in cat}
def enc(df):
    e = df.copy()
    for c in cat: e[c] = e[c].map({v: i for i, v in enumerate(CODES[c])}).astype(float)
    return e.astype(float)
def dec(a):
    e = pd.DataFrame(a, columns=num + cat)
    for c in cat: e[c] = e[c].round().astype(int).map(dict(enumerate(CODES[c])))
    return e
rs = np.random.RandomState(42); med = enc(X.loc[tr]).median()
bg = enc(X.loc[tr].sample(100, random_state=42)).fillna(med)
hi = np.where(pk >= np.quantile(pk, .9))[0]; lo = np.where(pk < np.quantile(pk, .9))[0]
idx = np.r_[rs.choice(hi, 150, replace=False), rs.choice(lo, 150, replace=False)]
XE = enc(X.loc[te].iloc[idx]).fillna(med)
sv = shap.PermutationExplainer(lambda a: gc.predict_proba(dec(a))[:, 1], shap.maskers.Independent(bg.values), max_evals=2 * XE.shape[1] + 1)(XE.values).values
NICE = {"age": "Age", "female": "Female", "educ_yrs": "Education (years)", "fam_income": "Family income", "fam_size": "Family size", "married": "Married",
        "employed": "Employed", "race": "Race/ethnicity", "pov_cat": "Income (% FPL)", "region": "Region"}
S = pd.DataFrame(sv, columns=num + cat); flag = np.r_[np.ones(150), np.zeros(150)].astype(bool)
imp = S.abs().mean().sort_values(ascending=False)
pd.DataFrame({"feature": [NICE[c] for c in imp.index], "mean_abs_shap": imp.values, "flagged": S[flag].mean()[imp.index].values,
              "not_flagged": S[~flag].mean()[imp.index].values}).round(5).to_csv("output/tables/knn_ses_shap.csv", index=False)
fig, AX = plt.subplots(2, 2, figsize=(12, 9))
a = AX[0, 0]
for wt, c in (("uniform", style.NAVY), ("distance", style.RUST)):
    s_ = cvr[cvr.param_weights == wt].sort_values("param_n_neighbors"); k = s_.param_n_neighbors.astype(int)
    a.plot(k, -s_.mean_test_score, "-o", ms=3.5, color=c, label=f"CV, {wt} weights"); a.fill_between(k, -s_.mean_test_score - s_.std_test_score, -s_.mean_test_score + s_.std_test_score, color=c, alpha=.12, lw=0)
a.set_xscale("log"); a.set_xticks([10, 25, 60, 150]); a.set_xticklabels(["10", "25", "60", "150"]); a.minorticks_off()
a.set_xlabel("Number of neighbours K (log scale)"); a.set_ylabel("CV mean squared error (log points²)"); a.set_title("Choosing K (demographics + SES only)"); a.legend(loc="upper right")
a = AX[0, 1]
for p_, c, l in ((pk, style.NAVY, f"KNN  AUC {auc_k:.2f}"), (pl, style.RUST, f"Logit  AUC {auc_l:.2f}")):
    fpr, tpr, _ = roc_curve(yc[te], p_, sample_weight=w[te]); a.plot(fpr, tpr, color=c, label=l)
a.plot([0, 1], [0, 1], "--", color=style.GRAY, lw=.8); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate"); a.set_title("High-cost users (top 10%), test set"); a.legend(loc="lower right")
top = imp.index[:10]; yy = np.arange(len(top))[::-1]
a = AX[1, 0]; a.barh(yy, 100 * imp[top], color=style.NAVY, height=.62); a.set_yticks(yy, [NICE[c] for c in top])
a.grid(axis="y", visible=False); a.grid(axis="x", visible=True); a.set_xlabel("Mean |SHAP| (pp of predicted high-cost probability)"); a.set_title("What the SES-only KNN relies on")
a = AX[1, 1]; m1, m0 = 100 * S[flag].mean()[top], 100 * S[~flag].mean()[top]
a.barh(yy + .18, m1, .36, color=style.RUST, label="Flagged high-cost"); a.barh(yy - .18, m0, .36, color=style.GRAY, label="Not flagged")
a.axvline(0, color="#333", lw=.8); a.set_yticks(yy, [NICE[c] for c in top]); a.grid(axis="y", visible=False); a.grid(axis="x", visible=True)
a.set_xlabel("Mean SHAP (pp; + pushes toward high-cost)"); a.set_title("What pushes adults into the high-cost flag")
a.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DDD", framealpha=1)
fig.tight_layout(); style.save(fig, "knn_ses_only", "Source: MEPS HC-243 (2022), adults 18+. KNN using demographics + SES only (no access or health). 5-fold CV on 80% training split; test metrics survey-weighted; "
                                "SHAP = permutation explainer on 300 test adults (150 flagged, 150 not), 100-person background.")
print(imp.round(4))
