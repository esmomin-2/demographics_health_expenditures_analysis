"""06 — Cross-validated K-nearest-neighbours prediction of healthcare expenditures.
Same predictors/split as 04_supervised (demographics + health status, no utilisation). Scaler inside the Pipeline;
K and weighting tuned by 5-fold CV (neg MSE). Benchmarks: training-mean (null) and OLS."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, roc_curve

d = pd.read_csv("data/meps2022_adults.csv")
NUM = ["age", "female", "educ_yrs", "fam_income", "fam_size", "health_poor", "mental_poor", "married", "employed",
       "has_usual_source", "diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer", "n_chronic"]
CAT = ["race", "pov_cat", "insurance", "region"]
pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
                         ("cat", Pipeline([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), ("sc", StandardScaler())]), CAT)])
X = d[NUM + CAT]; tr, te = train_test_split(d.index, test_size=0.2, random_state=42)
w_te = d.weight[te]; kf = KFold(n_splits=5, shuffle=True, random_state=42)
Ks = [1, 3, 5, 10, 15, 25, 40, 60, 80, 100, 150, 200, 300]
rows, curves = [], {}

def wm(y, p):  # weighted test metrics
    mse = mean_squared_error(y, p, sample_weight=w_te)
    return dict(test_MSE=mse, test_RMSE=np.sqrt(mse), test_MAE=mean_absolute_error(y, p, sample_weight=w_te), test_R2=r2_score(y, p, sample_weight=w_te))

for target in ["log_total_exp", "log_oop_exp"]:
    y = d[target]
    gs = GridSearchCV(Pipeline([("pre", pre), ("knn", KNeighborsRegressor())]),
                      {"knn__n_neighbors": Ks, "knn__weights": ["uniform", "distance"]},
                      cv=kf, scoring="neg_mean_squared_error", n_jobs=-1, return_train_score=True).fit(X.loc[tr], y[tr])
    cv = pd.DataFrame(gs.cv_results_); curves[target] = cv
    p = gs.predict(X.loc[te]); bk = gs.best_params_
    rows.append(dict(target=target, model=f"KNN (K={bk['knn__n_neighbors']}, {bk['knn__weights']})", cv_MSE=-gs.best_score_, **wm(y[te], p)))
    ols = Pipeline([("pre", pre), ("m", LinearRegression())]).fit(X.loc[tr], y[tr])
    rows.append(dict(target=target, model="OLS", cv_MSE=np.nan, **wm(y[te], ols.predict(X.loc[te]))))
    rows.append(dict(target=target, model="Null (training mean)", cv_MSE=np.nan, **wm(y[te], np.full(len(te), y[tr].mean()))))
    if target == "log_total_exp":
        # dollars: Duan smearing on the KNN log model
        resid = y[tr] - gs.predict(X.loc[tr]); smear = np.mean(np.exp(resid)); pd_ = np.expm1(p) * smear
        lev = dict(target="total_exp ($, smeared)", model=rows[-3]["model"], cv_MSE=np.nan, **wm(d.total_exp[te], pd_))
        best_tot = gs

rows.append(lev); res = pd.DataFrame(rows); res.round(4).to_csv("output/tables/k_knn_metrics.csv", index=False)
print(res.round(3).to_string())

# high-cost (top 10%, weighted) classification with KNN
cut = np.quantile(np.repeat(d.total_exp.values, 1), 0.9)
o = d.total_exp.sort_values(); cwt = d.weight[o.index].cumsum() / d.weight.sum(); cut = o[cwt >= 0.9].iloc[0]
yc = (d.total_exp >= cut).astype(int)
gc = GridSearchCV(Pipeline([("pre", pre), ("knn", KNeighborsClassifier())]), {"knn__n_neighbors": Ks[2:], "knn__weights": ["uniform", "distance"]},
                  cv=kf, scoring="roc_auc", n_jobs=-1).fit(X.loc[tr], yc[tr])
pk = gc.predict_proba(X.loc[te])[:, 1]
lg = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=2000))]).fit(X.loc[tr], yc[tr]); pl = lg.predict_proba(X.loc[te])[:, 1]
auc_k, auc_l = roc_auc_score(yc[te], pk, sample_weight=w_te), roc_auc_score(yc[te], pl, sample_weight=w_te)
pd.DataFrame([dict(model=f"KNN (K={gc.best_params_['knn__n_neighbors']}, {gc.best_params_['knn__weights']})", cv_AUC=gc.best_score_, test_AUC=auc_k),
              dict(model="Logit", cv_AUC=np.nan, test_AUC=auc_l)]).round(4).to_csv("output/tables/k_knn_highcost.csv", index=False)
print(f"High-cost cutoff ${cut:,.0f}; KNN AUC {auc_k:.3f} (K={gc.best_params_}), logit AUC {auc_l:.3f}")

# ---------------- figures
fig, AX = plt.subplots(2, 2, figsize=(12, 9)); ax = AX[0]
for a, (target, lab) in zip(ax[:1], [("log_total_exp", "Choosing K: cross-validated error")]):
    cv = curves[target]
    for wt, c in (("uniform", style.NAVY), ("distance", style.RUST)):
        s = cv[cv.param_knn__weights == wt].sort_values("param_knn__n_neighbors")
        k = s.param_knn__n_neighbors.astype(int); m = -s.mean_test_score; sd = s.std_test_score
        a.plot(k, m, "-o", ms=3.5, color=c, label=f"CV, {wt} weights"); a.fill_between(k, m - sd, m + sd, color=c, alpha=.12, lw=0)
        if wt == "uniform": a.plot(k, -s.mean_train_score, "--", color=c, lw=1, label="Training error")
    nul = res[(res.target == target) & (res.model.str.startswith("Null"))].test_MSE.iloc[0]
    a.axhline(nul, color=style.GRAY, lw=.8, ls=":"); a.text(300, nul, "null model", ha="right", va="bottom", fontsize=8, color=style.GRAY)
    a.set_xscale("log"); a.set_xticks([1,3,10,30,100,300]); a.set_xticklabels(["1","3","10","30","100","300"]); a.minorticks_off(); a.set_xlabel("Number of neighbours K (log scale)"); a.set_ylabel("Mean squared error (log points²)"); a.set_title(lab)
ax[0].legend(loc="upper right")
fpr, tpr, _ = roc_curve(yc[te], pk, sample_weight=w_te); ax[1].plot(fpr, tpr, color=style.NAVY, label=f"KNN  AUC {auc_k:.2f}")
fpr, tpr, _ = roc_curve(yc[te], pl, sample_weight=w_te); ax[1].plot(fpr, tpr, color=style.RUST, label=f"Logit  AUC {auc_l:.2f}")
ax[1].plot([0, 1], [0, 1], color=style.GRAY, lw=.8, ls="--"); ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].set_title("High-cost users (top 10%), test set"); ax[1].legend(loc="lower right")
# ---------------- explainability: SHAP (model-agnostic permutation explainer) for the KNN high-cost classifier
import shap
CODES = {c: sorted(d[c].dropna().unique()) for c in CAT}
def enc(df):
    e = df.copy()
    for c in CAT: e[c] = e[c].map({v: i for i, v in enumerate(CODES[c])}).astype(float)
    return e.astype(float)
def dec(a):
    e = pd.DataFrame(a, columns=NUM + CAT)
    for c in CAT: e[c] = e[c].round().astype(int).map(dict(enumerate(CODES[c])))
    return e
f_ = lambda a: gc.predict_proba(dec(a))[:, 1]
rs = np.random.RandomState(42)
bg = enc(X.loc[tr].sample(100, random_state=42)).fillna(enc(X.loc[tr]).median())
pt = gc.predict_proba(X.loc[te])[:, 1]
ex_idx = np.r_[rs.choice(np.where(pt >= np.quantile(pt, .9))[0], 150, replace=False), rs.choice(np.where(pt < np.quantile(pt, .9))[0], 150, replace=False)]
XE = enc(X.loc[te].iloc[ex_idx]).fillna(enc(X.loc[tr]).median())
ex = shap.PermutationExplainer(f_, shap.maskers.Independent(bg.values), max_evals=2 * XE.shape[1] + 1)
sv = ex(XE.values).values
NICE = {"age": "Age", "female": "Female", "educ_yrs": "Education (yrs)", "fam_income": "Family income", "fam_size": "Family size",
        "health_poor": "Poor self-rated health", "mental_poor": "Poor mental health", "married": "Married", "employed": "Employed",
        "has_usual_source": "Usual source of care", "diabetes": "Diabetes", "hypertension": "Hypertension", "heart_disease": "Heart disease",
        "asthma": "Asthma", "arthritis": "Arthritis", "cancer": "Cancer", "n_chronic": "No. chronic conditions",
        "race": "Race/ethnicity", "pov_cat": "Income (% FPL)", "insurance": "Insurance", "region": "Region"}
S = pd.DataFrame(sv, columns=NUM + CAT); flag = np.r_[np.ones(150), np.zeros(150)].astype(bool)
imp = S.abs().mean().sort_values(ascending=False); top = imp.index[:10]
pd.DataFrame({"feature": [NICE[c] for c in imp.index], "mean_abs_shap": imp.values,
              "mean_shap_flagged": S[flag].mean()[imp.index].values, "mean_shap_not_flagged": S[~flag].mean()[imp.index].values}
             ).round(5).to_csv("output/tables/k_knn_shap_highcost.csv", index=False)
a = AX[1, 0]; yy = np.arange(len(top))[::-1]
a.barh(yy, 100 * imp[top], color=style.NAVY, height=.62); a.set_yticks(yy, [NICE[c] for c in top]); a.grid(axis="y", visible=False); a.grid(axis="x", visible=True)
a.set_xlabel("Mean |SHAP| (pp of predicted high-cost probability)"); a.set_title("What the KNN relies on")
a = AX[1, 1]; m1, m0 = 100 * S[flag].mean()[top], 100 * S[~flag].mean()[top]
a.barh(yy + .18, m1, .36, color=style.RUST, label="Flagged high-cost (top 10% predicted)"); a.barh(yy - .18, m0, .36, color=style.GRAY, label="Not flagged")
a.axvline(0, color="#333", lw=.8); a.set_yticks(yy, [NICE[c] for c in top]); a.grid(axis="y", visible=False); a.grid(axis="x", visible=True)
a.set_xlabel("Mean SHAP (pp; + pushes toward high-cost)"); a.set_title("What pushes adults into the high-cost flag")
a.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=1, fontsize=8.5)
print(pd.DataFrame({"abs": imp[top], "flagged": S[flag].mean()[top], "not": S[~flag].mean()[top]}).round(4))
fig.tight_layout(); style.save(fig, "k_knn_cv", "Source: MEPS HC-243 (2022), adults 18+. 5-fold CV on 80% training split; shaded = ±1 SD across folds; test metrics survey-weighted. SHAP: permutation explainer on 300 test adults (150 flagged, 150 not), 100-person background.")

# predicted vs actual (test) for log total
p = best_tot.predict(X.loc[te]); y = d.log_total_exp[te]
fig, a = plt.subplots(figsize=(5.2, 4.6)); a.scatter(p, y, s=4, alpha=.3, color=style.NAVY, lw=0)
a.plot([0, 13], [0, 13], color=style.RUST, lw=1); a.set_xlabel("KNN predicted log(1 + $)"); a.set_ylabel("Actual log(1 + $)")
a.set_title(f"KNN predictions on test set (R² = {r2_score(y, p, sample_weight=w_te):.2f})"); a.grid(axis="x", visible=True)
fig.tight_layout(); style.save(fig, "k_knn_pred_vs_actual")
