"""04_supervised.py - Supervised ML for MEPS 2022 adult health spending.
Predictors: demographics + health status only (no utilization / spending).
Sections A-F: regression, two-part model, high-cost classification, OOP,
interpretation (LASSO path, permutation importance, SHAP), fairness check."""
import os, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.ticker
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import (LinearRegression, LassoCV, RidgeCV, ElasticNetCV,
                                  LogisticRegression, LogisticRegressionCV, lasso_path)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,
                             brier_score_loss, accuracy_score, precision_score, recall_score, roc_curve)
from xgboost import XGBRegressor, XGBClassifier
import shap
import sys; sys.path.insert(0, "code"); import style; style.apply()
warnings.filterwarnings("ignore")
RS = 42; t0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = os.path.join(ROOT, "output/figures"), os.path.join(ROOT, "output/tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True); os.chdir(ROOT)

df = pd.read_csv(os.path.join(ROOT, "data/meps2022_adults.csv"))
NUM = ["age", "female", "educ_yrs", "fam_income", "fam_size", "health_poor", "mental_poor", "married",
       "employed", "has_usual_source", "diabetes", "hypertension", "heart_disease", "asthma",
       "arthritis", "cancer", "n_chronic"]
CAT = ["race", "pov_cat", "insurance", "region"]
X_raw = pd.get_dummies(df[NUM + CAT], columns=CAT, drop_first=False, dtype=float)
X_raw.columns = [c.replace('<', 'lt').replace('[', '(').replace(']', ')') for c in X_raw.columns]
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RS)
med = X_raw.iloc[idx_tr].median()
X = X_raw.fillna(med)
# drop one reference level per categorical for linear models (avoid collinearity)
REF = ["race_NH White", "pov_cat_High income", "insurance_Any private", "region_South"]
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
def pipe(m): return Pipeline([("scaler", StandardScaler()), ("model", m)])
SW = "model__sample_weight"
Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
w = df["weight"].values; wtr, wte = w[idx_tr], w[idx_te]
# linear models / MLP use the reference-dropped design; StandardScaler lives inside each Pipeline
Ztr, Zte = Xtr.drop(columns=REF), Xte.drop(columns=REF)
y = df["log_total_exp"].values; ytr, yte = y[idx_tr], y[idx_te]
lvl = df["total_exp"].values; lvl_te = lvl[idx_te]
cv = KFold(5, shuffle=True, random_state=RS)

def wmet(yt, yp, ww):
    return dict(MSE=mean_squared_error(yt, yp, sample_weight=ww), RMSE=np.sqrt(mean_squared_error(yt, yp, sample_weight=ww)),
                MAE=mean_absolute_error(yt, yp, sample_weight=ww), R2=r2_score(yt, yp, sample_weight=ww))

# ---------------- A. one-part regression on log_total_exp ----------------
print(f"[{(time.time()-t0)/60:.1f}m] A. regression"); models, preds = {}, {}
models["OLS"] = pipe(LinearRegression()).fit(Ztr, ytr, **{SW: wtr})
models["LASSO"] = pipe(LassoCV(cv=cv, random_state=RS, n_alphas=60)).fit(Ztr, ytr, **{SW: wtr})
models["Ridge"] = pipe(RidgeCV(alphas=np.logspace(-3, 4, 40), cv=cv, scoring="neg_mean_squared_error")).fit(Ztr, ytr, **{SW: wtr})
models["ElasticNet"] = pipe(ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95], cv=cv, random_state=RS,
                                    n_alphas=40)).fit(Ztr, ytr, **{SW: wtr})
for k in ["OLS", "LASSO", "Ridge", "ElasticNet"]: preds[k] = models[k].predict(Zte)
rf_s = RandomizedSearchCV(RandomForestRegressor(n_estimators=300, random_state=RS, n_jobs=-1),
    {"max_depth": [6, 10, 14, None], "min_samples_leaf": [5, 10, 20, 50], "max_features": [0.3, 0.5, "sqrt"]},
    n_iter=8, cv=cv, scoring="neg_mean_squared_error", random_state=RS, n_jobs=1)
rf_s.fit(Xtr, ytr, sample_weight=wtr); models["RandomForest"] = rf_s.best_estimator_
xgb_grid = {"n_estimators": [200, 400, 700], "max_depth": [2, 3, 4, 6], "learning_rate": [0.02, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 20],
            "reg_lambda": [0.5, 1, 5]}
xgb_s = RandomizedSearchCV(XGBRegressor(random_state=RS, n_jobs=4, tree_method="hist"), xgb_grid, n_iter=25,
                           cv=cv, scoring="neg_mean_squared_error", random_state=RS)
xgb_s.fit(Xtr, ytr, sample_weight=wtr); models["XGBoost"] = xgb_s.best_estimator_
gb_s = RandomizedSearchCV(GradientBoostingRegressor(random_state=RS),
    {"n_estimators": [200, 400], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.05, 0.1],
     "subsample": [0.7, 1.0], "min_samples_leaf": [10, 30]}, n_iter=8, cv=cv,
    scoring="neg_mean_squared_error", random_state=RS, n_jobs=-1)
gb_s.fit(Xtr, ytr, sample_weight=wtr); models["GradBoost"] = gb_s.best_estimator_
mlp_s = RandomizedSearchCV(pipe(MLPRegressor(max_iter=500, early_stopping=True, random_state=RS)),
    {"model__hidden_layer_sizes": [(32,), (64, 32), (128, 64)], "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
     "model__learning_rate_init": [1e-3, 3e-3]}, n_iter=6, cv=cv, scoring="neg_mean_squared_error", random_state=RS)
mlp_s.fit(Ztr, ytr)  # MLPRegressor.fit has no sample_weight in sklearn 1.6
models["MLP"] = mlp_s.best_estimator_
preds["RandomForest"] = models["RandomForest"].predict(Xte); preds["XGBoost"] = models["XGBoost"].predict(Xte); preds["GradBoost"] = models["GradBoost"].predict(Xte)
preds["MLP"] = models["MLP"].predict(Zte)

def smear_levels(model, Xa_tr, Xa_te):
    res = ytr - model.predict(Xa_tr)
    S = np.average(np.exp(res), weights=wtr)  # Duan smearing factor
    return np.exp(model.predict(Xa_te)) * S, S
rows = []
for k, p in preds.items():
    m = wmet(yte, p, wte); Xa = (Xtr, Xte) if k in ["RandomForest", "XGBoost", "GradBoost"] else (Ztr, Zte)
    lv, S = smear_levels(models[k], *Xa)
    m.update(RMSE_dollars=np.sqrt(mean_squared_error(lvl_te, lv, sample_weight=wte)),
             MAE_dollars=mean_absolute_error(lvl_te, lv, sample_weight=wte),
             R2_levels_smeared=r2_score(lvl_te, lv, sample_weight=wte), smear_factor=S)
    rows.append(dict(model=k, **m))
reg = pd.DataFrame(rows).sort_values("RMSE"); reg.to_csv(f"{TAB}/s_regression_metrics.csv", index=False)
print(reg.round(3).to_string()); best = reg.iloc[0]["model"]
pd.DataFrame({"param": list(xgb_s.best_params_), "value": list(xgb_s.best_params_.values())}).to_csv(
    f"{TAB}/s_xgb_best_params.csv", index=False)

# ---------------- B. two-part model ----------------
print(f"[{(time.time()-t0)/60:.1f}m] B. two-part"); d = df["any_exp"].values; dtr, dte = d[idx_tr], d[idx_te]
clf = {}
clf["Logit"] = pipe(LogisticRegression(C=1e6, max_iter=2000)).fit(Ztr, dtr, **{SW: wtr})
clf["LASSO-logit"] = pipe(LogisticRegressionCV(Cs=20, cv=cv, penalty="l1", solver="saga", scoring="neg_log_loss",
                                          max_iter=3000, random_state=RS)).fit(Ztr, dtr, **{SW: wtr})
clf["RandomForest"] = RandomizedSearchCV(RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1),
    {"max_depth": [6, 10, None], "min_samples_leaf": [5, 20, 50], "max_features": [0.3, "sqrt"]},
    n_iter=6, cv=cv, scoring="roc_auc", random_state=RS).fit(Xtr, dtr, sample_weight=wtr).best_estimator_
clf["XGBoost"] = RandomizedSearchCV(XGBClassifier(random_state=RS, n_jobs=4, tree_method="hist"), xgb_grid,
    n_iter=15, cv=cv, scoring="roc_auc", random_state=RS).fit(Xtr, dtr, sample_weight=wtr).best_estimator_
crow, pany = [], {}
for k, m in clf.items():
    Xa = Zte if k in ["Logit", "LASSO-logit"] else Xte
    p = m.predict_proba(Xa)[:, 1]; pany[k] = p
    crow.append(dict(model=k, AUC=roc_auc_score(dte, p, sample_weight=wte),
                     Brier=brier_score_loss(dte, p, sample_weight=wte),
                     Accuracy=accuracy_score(dte, p > 0.5, sample_weight=wte)))
cls = pd.DataFrame(crow); cls.to_csv(f"{TAB}/s_part1_any_exp_metrics.csv", index=False); print(cls.round(3))
pos = dtr == 1
x2 = RandomizedSearchCV(XGBRegressor(random_state=RS, n_jobs=4, tree_method="hist"), xgb_grid, n_iter=15, cv=cv,
                        scoring="neg_mean_squared_error", random_state=RS)
x2.fit(Xtr[pos], ytr[pos], sample_weight=wtr[pos]); m2 = x2.best_estimator_
S2 = np.average(np.exp(ytr[pos] - m2.predict(Xtr[pos])), weights=wtr[pos])
cond = np.exp(m2.predict(Xte)) * S2
pos_te = dte == 1
p2met = wmet(yte[pos_te], m2.predict(Xte[pos_te]), wte[pos_te])
best_clf = cls.sort_values("AUC", ascending=False).iloc[0]["model"]
tp = []
for k in clf:
    e = pany[k] * cond
    tp.append(dict(model=f"Two-part ({k} + XGB)", RMSE_dollars=np.sqrt(mean_squared_error(lvl_te, e, sample_weight=wte)),
                   MAE_dollars=mean_absolute_error(lvl_te, e, sample_weight=wte), R2_levels=r2_score(lvl_te, e, sample_weight=wte)))
for _, r in reg.iterrows():
    tp.append(dict(model=f"One-part {r['model']} (smeared)", RMSE_dollars=r["RMSE_dollars"],
                   MAE_dollars=r["MAE_dollars"], R2_levels=r["R2_levels_smeared"]))
tpm = pd.DataFrame(tp); tpm.to_csv(f"{TAB}/s_twopart_vs_onepart_dollars.csv", index=False)
print("part2 (positives, log scale):", {k: round(v, 3) for k, v in p2met.items()}); print(tpm.round(3))

# ---------------- C. high-cost classification ----------------
print(f"[{(time.time()-t0)/60:.1f}m] C. high-cost")
o = np.argsort(lvl); cw = np.cumsum(w[o]) / w.sum()
thr = lvl[o][np.searchsorted(cw, 0.90)]
hc = (lvl > thr).astype(int); htr, hte = hc[idx_tr], hc[idx_te]
hx = RandomizedSearchCV(XGBClassifier(random_state=RS, n_jobs=4, tree_method="hist"), xgb_grid, n_iter=15, cv=cv,
                        scoring="roc_auc", random_state=RS).fit(Xtr, htr, sample_weight=wtr).best_estimator_
hl = pipe(LogisticRegression(C=1e6, max_iter=3000)).fit(Ztr, htr, **{SW: wtr})
hrow = []; fig = plt.figure(figsize=(5.5, 5)); cols_ = iter([style.NAVY, style.RUST])
for k, p in [("XGBoost", hx.predict_proba(Xte)[:, 1]), ("Logit", hl.predict_proba(Zte)[:, 1])]:
    cut = np.quantile(p, 0.9)  # flag top 10% predicted risk
    hrow.append(dict(model=k, threshold_dollars=thr, AUC=roc_auc_score(hte, p, sample_weight=wte),
        Precision_top10=precision_score(hte, p >= cut, sample_weight=wte), Recall_top10=recall_score(hte, p >= cut, sample_weight=wte),
        Precision_p50=precision_score(hte, p >= .5, sample_weight=wte, zero_division=0),
        Recall_p50=recall_score(hte, p >= .5, sample_weight=wte)))
    fpr, tpr, _ = roc_curve(hte, p, sample_weight=wte); plt.plot(fpr, tpr, color=next(cols_), label=f"{k} (AUC={hrow[-1]['AUC']:.3f})")
plt.plot([0, 1], [0, 1], "--", color=style.GRAY, lw=.8); plt.xlabel("False positive rate (%)"); plt.ylabel("True positive rate (%)")
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
plt.title(f"ROC: high-cost users (>${thr:,.0f})"); plt.legend(); style.save(fig, "s_roc_highcost")
hcm = pd.DataFrame(hrow); hcm.to_csv(f"{TAB}/s_highcost_metrics.csv", index=False); print(hcm.round(3))

# ---------------- D. OOP ----------------
print(f"[{(time.time()-t0)/60:.1f}m] D. OOP"); yo = df["log_oop_exp"].values
xo = RandomizedSearchCV(XGBRegressor(random_state=RS, n_jobs=4, tree_method="hist"), xgb_grid, n_iter=15, cv=cv,
                        scoring="neg_mean_squared_error", random_state=RS).fit(Xtr, yo[idx_tr], sample_weight=wtr).best_estimator_
ol = pipe(LinearRegression()).fit(Ztr, yo[idx_tr], **{SW: wtr})
oop = pd.DataFrame([dict(model="XGBoost", **wmet(yo[idx_te], xo.predict(Xte), wte)),
                    dict(model="OLS", **wmet(yo[idx_te], ol.predict(Zte), wte))])
oop.to_csv(f"{TAB}/s_oop_metrics.csv", index=False); print(oop.round(3))

# ---------------- E. interpretation ----------------
print(f"[{(time.time()-t0)/60:.1f}m] E. interpretation")
Zs = models["LASSO"].named_steps["scaler"].transform(Ztr)
alphas, coefs, _ = lasso_path(Zs * np.sqrt(wtr / wtr.mean())[:, None],
                              (ytr - np.average(ytr, weights=wtr)) * np.sqrt(wtr / wtr.mean()), n_alphas=80)
fig = plt.figure(figsize=(8, 5.5)); top = np.argsort(-np.abs(models["LASSO"][-1].coef_))[:8]
for j in range(coefs.shape[0]):
    plt.plot(np.log10(alphas), coefs[j], lw=1.6 if j in top else .6, color=style.PALETTE[list(top).index(j)] if j in top else "#D0D0D0",
             label=Ztr.columns[j] if j in top else None)
plt.axvline(np.log10(models["LASSO"][-1].alpha_), ls="--", c=style.GRAY, lw=.8, label="CV alpha")
plt.xlabel("Penalty, log10(alpha) (log10 units)"); plt.ylabel("Coefficient (log points per SD)"); plt.title("LASSO path for log total spending")
plt.legend(fontsize=7); style.save(fig, "s_lasso_path")
lc = pd.DataFrame({"feature": Ztr.columns, "coef_std": models["LASSO"][-1].coef_})
lc = lc[lc.coef_std != 0].reindex(lc.coef_std.abs().sort_values(ascending=False).index).dropna()
lc.to_csv(f"{TAB}/s_lasso_nonzero_coefs.csv", index=False)
print(f"LASSO alpha={models['LASSO'][-1].alpha_:.4f}, nonzero={len(lc)}/{Ztr.shape[1]}")

pi_rows = []
for k in ["RandomForest", "XGBoost"]:
    r = permutation_importance(models[k], Xte, yte, n_repeats=5, random_state=RS, sample_weight=wte, n_jobs=1)
    pi_rows.append(pd.DataFrame({"model": k, "feature": Xte.columns, "importance_mean": r.importances_mean,
                                 "importance_sd": r.importances_std}))
pi = pd.concat(pi_rows); pi.to_csv(f"{TAB}/s_permutation_importance.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for a, k in zip(ax, ["RandomForest", "XGBoost"]):
    s = pi[pi.model == k].nlargest(15, "importance_mean")[::-1]
    a.barh(s.feature, s.importance_mean, xerr=s.importance_sd, color=style.NAVY); a.set_title(f"{k} permutation importance"); a.set_xlabel("Drop in test R² (R² points)")
plt.tight_layout(); style.save(fig, "s_permutation_importance")

rng = np.random.RandomState(RS); sidx = rng.choice(len(Xte), min(3000, len(Xte)), replace=False)
Xs = Xte.iloc[sidx]; ws = wte[sidx]
expl = shap.TreeExplainer(models["XGBoost"]); sv = expl.shap_values(Xs)
shap.summary_plot(sv, Xs, show=False, max_display=20, cmap=plt.get_cmap(style.CMAP)); plt.title("SHAP values, XGBoost")
plt.xlabel("SHAP value (log points of total $)"); style.save(plt.gcf(), "s_shap_beeswarm")
shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=20, color=style.NAVY); plt.title("Mean |SHAP|, XGBoost")
plt.xlabel("Mean |SHAP| (log points of total $)"); style.save(plt.gcf(), "s_shap_bar")
for f in ["age", "fam_income", "n_chronic"]:
    shap.dependence_plot(f, sv, Xs, show=False, interaction_index="auto", cmap=plt.get_cmap(style.CMAP))
    unit = {"age": "years", "fam_income": "$", "n_chronic": "count"}[f]
    plt.xlabel(f"{f} ({unit})"); plt.ylabel(f"SHAP for {f} (log points)"); plt.title(f"SHAP dependence: {f}")
    style.save(plt.gcf(), f"s_shap_dependence_{f}")
svdf = pd.DataFrame(sv, columns=Xs.columns)
grp = {c: next((g for g in CAT if c.startswith(g + "_")), c) for c in Xs.columns}
agg = svdf.T.groupby(pd.Series(grp)).sum().T  # sum one-hot SHAP within group, then |.|
ms = pd.DataFrame({"feature": agg.columns, "wmean_abs_shap": [np.average(np.abs(agg[c]), weights=ws) for c in agg]})
# direction: weighted corr of feature value with SHAP (numeric only)
dirs = {}
for c in NUM:
    v = Xs[c].values; dirs[c] = np.sign(np.cov(v, sv[:, list(Xs.columns).index(c)], aweights=ws)[0, 1])
ms["direction_(+:higher_value_more_spend)"] = ms.feature.map(dirs)
ms = ms.sort_values("wmean_abs_shap", ascending=False); ms.to_csv(f"{TAB}/s_shap_mean_abs.csv", index=False)
print(ms.round(3).to_string())
gm = []
for g in ["race", "insurance", "pov_cat"]:
    cols = [c for c in Xs.columns if c.startswith(g + "_")]
    lab = df[g].values[idx_te][sidx]
    for L in np.unique(lab):
        m = lab == L
        gm.append(dict(group=g, level=L, n=m.sum(), wmean_shap_group_dummies=np.average(agg[g].values[m], weights=ws[m]),
                       wmean_total_shap=np.average(sv[m].sum(1), weights=ws[m])))
gm = pd.DataFrame(gm); gm.to_csv(f"{TAB}/s_shap_group_means.csv", index=False); print(gm.round(3))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, g in zip(ax, ["race", "insurance"]):
    s = gm[gm.group == g]; a.barh(s.level, s.wmean_shap_group_dummies, color=style.TEAL)
    a.axvline(0, c=style.GRAY, lw=.7); a.set_title(f"Mean SHAP by {g}"); a.set_xlabel("Mean SHAP (log points of total $)")
plt.tight_layout(); style.save(fig, "s_shap_group_means")

# ---------------- F. fairness ----------------
print(f"[{(time.time()-t0)/60:.1f}m] F. fairness"); fr = []
for g in ["race", "pov_cat"]:
    lab = df[g].values[idx_te]
    for L in np.unique(lab):
        m = lab == L
        for k in ["OLS", "XGBoost"]:
            fr.append(dict(group=g, level=L, model=k, n=m.sum(), **wmet(yte[m], preds[k][m], wte[m]),
                           mean_resid=np.average(yte[m] - preds[k][m], weights=wte[m])))
fr = pd.DataFrame(fr); fr.to_csv(f"{TAB}/s_fairness_rmse_by_group.csv", index=False)
print(fr[fr.model == "XGBoost"].round(3).to_string())
# ---------------- LMG / Shapley R2 decomposition (OLS, grouped blocks) ----------------
print("LMG"); from itertools import combinations; from math import factorial
blocks = {"age": ["age"], "sex": ["female"], "race": [c for c in Ztr if c.startswith("race_")],
    "income/poverty": ["fam_income"] + [c for c in Ztr if c.startswith("pov_cat_")], "education": ["educ_yrs"],
    "family": ["married", "fam_size"], "region": [c for c in Ztr if c.startswith("region_")],
    "insurance": [c for c in Ztr if c.startswith("insurance_")],
    "health": ["health_poor", "mental_poor", "diabetes", "hypertension", "heart_disease", "asthma", "arthritis",
               "cancer", "n_chronic", "has_usual_source", "employed"]}
B = list(blocks); r2c = {(): 0.0}
for k in range(1, len(B) + 1):
    for S in combinations(B, k):
        cols = sum((blocks[b] for b in S), [])
        r2c[S] = LinearRegression().fit(Xtr[cols], ytr, sample_weight=wtr).score(Xtr[cols], ytr, sample_weight=wtr)
P = len(B); lmg = {}
for b in B:
    oth = [x for x in B if x != b]; v = 0
    for k in range(P):
        for S in combinations(oth, k):
            Sb = tuple(x for x in B if x in S or x == b)
            v += factorial(k) * factorial(P - k - 1) / factorial(P) * (r2c[Sb] - r2c[S])
    lmg[b] = v
lm = pd.DataFrame({"block": B, "lmg_r2": [lmg[b] for b in B]}).sort_values("lmg_r2", ascending=False)
lm["share_of_r2"] = lm.lmg_r2 / lm.lmg_r2.sum(); lm.to_csv(f"{TAB}/s_lmg.csv", index=False)
print(lm.round(4).to_string(), "\nfull R2 (train, weighted):", round(r2c[tuple(B)], 4))
print(f"Best one-part model: {best}; total runtime {(time.time()-t0)/60:.1f} min")
