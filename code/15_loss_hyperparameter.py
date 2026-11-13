"""15 — Loss-function hyperparameter (ECON442 Sep 2 notes): choose a in |y - x'θ|^a by K-fold CV.
a = 1 -> LAD (conditional median), a = 2 -> OLS (conditional mean). Outcome log(1+total $); predictors = demographics, SES,
access, health. Survey-weighted objective. CV criterion = validation MAE in log points (robust to the long tail) and MSE.
Then compare race/age/sex gaps at a = 1 (median) vs a = 2 (mean)."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split
d = pd.read_csv("data/meps2022_adults.csv")
d["age_grp"] = pd.cut(d.age, [17, 34, 49, 64, 120], labels=["18-34", "35-49", "50-64", "65+"]).astype(str)
d = d.dropna(subset=["health_poor", "mental_poor", "educ_yrs", "employed", "married", "has_usual_source", "n_chronic"])
X = pd.get_dummies(d[["age_grp", "race", "pov_cat", "insurance", "region"]], drop_first=False, dtype=float)
X = X.drop(columns=["age_grp_18-34", "race_NH White", "pov_cat_Middle income", "insurance_Any private", "region_South"])
X = pd.concat([X, d[["female", "educ_yrs", "employed", "married", "fam_size", "has_usual_source", "health_poor", "mental_poor", "n_chronic"]]], axis=1)
mu, sd = X.mean(), X.std().replace(0, 1); Z = ((X - mu) / sd).values; Z = np.c_[np.ones(len(Z)), Z]
y = d.log_total_exp.values; w = (d.weight / d.weight.mean()).values

def fit(Zt, yt, wt, a, eps=1e-6):
    b0 = np.linalg.lstsq(Zt * np.sqrt(wt)[:, None], yt * np.sqrt(wt), rcond=None)[0]
    if a == 2: return b0
    f = lambda b: np.sum(wt * (np.sqrt((yt - Zt @ b) ** 2 + eps)) ** a) / len(yt)            # smoothed |r|^a (kink at 0 for a=1)
    g = lambda b: -Zt.T @ (wt * a * (np.sqrt((yt - Zt @ b) ** 2 + eps)) ** (a - 2) * (yt - Zt @ b)) / len(yt)
    return minimize(f, b0, jac=g, method="L-BFGS-B", options={"maxiter": 3000}).x
A = [1.0, 1.25, 1.5, 1.75, 2.0]
tr, te = train_test_split(np.arange(len(y)), test_size=.2, random_state=42)
kf = KFold(5, shuffle=True, random_state=42); rows = []
for a in A:
    mae, mse = [], []
    for i, j in kf.split(tr):
        b = fit(Z[tr[i]], y[tr[i]], w[tr[i]], a); r = y[tr[j]] - Z[tr[j]] @ b
        mae.append(np.average(np.abs(r), weights=w[tr[j]])); mse.append(np.average(r ** 2, weights=w[tr[j]]))
    rows.append(dict(a=a, cv_MAE=np.mean(mae), cv_MAE_sd=np.std(mae), cv_MSE=np.mean(mse), cv_MSE_sd=np.std(mse)))
CV = pd.DataFrame(rows); a_mae, a_mse = CV.loc[CV.cv_MAE.idxmin(), "a"], CV.loc[CV.cv_MSE.idxmin(), "a"]
for a in A:
    b = fit(Z[tr], y[tr], w[tr], a); r = y[te] - Z[te] @ b
    CV.loc[CV.a == a, "test_MAE"] = np.average(np.abs(r), weights=w[te]); CV.loc[CV.a == a, "test_MSE"] = np.average(r ** 2, weights=w[te])
CV.round(4).to_csv("output/tables/loss_cv.csv", index=False); print(CV.round(3)); print("a* (MAE):", a_mae, " a* (MSE):", a_mse)

# coefficients at a=1 and a=2 on full sample, bootstrap CIs (PSU-cluster bootstrap within strata)
KEY = {"race_NH Black": "NH Black vs NH White", "race_Hispanic": "Hispanic vs NH White", "race_NH Asian": "NH Asian vs NH White",
       "age_grp_65+": "Age 65+ vs 18-34", "age_grp_50-64": "Age 50-64 vs 18-34", "female": "Female vs male",
       "insurance_Uninsured": "Uninsured vs private", "pov_cat_High income": "High vs middle income", "has_usual_source": "Usual source of care"}
cols = ["const"] + list(X.columns); idx = [cols.index(k) for k in KEY]
def unstd(b, k): c = cols[k]; return b[k] / sd[c]          # per-unit (dummy: 0->1) effect
est = {a: fit(Z, y, w, a) for a in (1.0, 2.0)}
g = d.strata.astype(str) + "_" + d.psu.astype(str); grp = pd.Series(np.arange(len(d))).groupby(g.values).apply(list)
rng = np.random.default_rng(42); B = 200; boot = {1.0: [], 2.0: []}
st = pd.Series(grp.index.str.split("_").str[0], index=grp.index)
for _ in range(B):
    pick = []
    for s, ps in st.groupby(st).groups.items():
        ps = list(ps); pick += [i for p in rng.choice(ps, len(ps)) for i in grp[p]]
    pick = np.array(pick)
    for a in (1.0, 2.0): boot[a].append(fit(Z[pick], y[pick], w[pick], a))
out = []
for a in (1.0, 2.0):
    bb = np.array(boot[a])
    for k, lab in zip(idx, KEY.values()):
        e = unstd(est[a], k); dist = bb[:, k] / sd[cols[k]]
        out.append(dict(a=a, term=lab, coef=e, lo=np.percentile(dist, 2.5), hi=np.percentile(dist, 97.5)))
C = pd.DataFrame(out); C["pct"], C["pct_lo"], C["pct_hi"] = [100 * (np.exp(C[c]) - 1) for c in ["coef", "lo", "hi"]]
C.round(4).to_csv("output/tables/loss_lad_vs_ols.csv", index=False); print(C.round(1).to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[1, 1.5]))
ax[0].errorbar(CV.a, CV.cv_MAE, yerr=CV.cv_MAE_sd, fmt="-o", color=style.NAVY, capsize=3, label="CV MAE")
ax[0].set_xlabel("Loss exponent a in |y − x′θ|^a"); ax[0].set_ylabel("5-fold CV mean absolute error (log points)")
ax2 = ax[0].twinx(); ax2.errorbar(CV.a, CV.cv_MSE, yerr=CV.cv_MSE_sd, fmt="--s", color=style.RUST, capsize=3, label="CV MSE"); ax2.set_ylabel("CV mean squared error (log points²)", color=style.RUST)
ax2.spines["right"].set_visible(True); ax2.grid(False)
ax[0].set_xticks(A, ["1\n(LAD)", "1.25", "1.5", "1.75", "2\n(OLS)"]); ax[0].set_title("Choosing the loss by cross-validation")
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[0].legend(h1 + h2, l1 + l2, loc="upper center")
yy = np.arange(len(KEY))[::-1]
for k, (a, c, lab) in enumerate([(1.0, style.TEAL, "a = 1: median (LAD)"), (2.0, style.NAVY, "a = 2: mean (OLS)")]):
    s = C[C.a == a].set_index("term").loc[list(KEY.values())]
    ax[1].errorbar(s.pct, yy + (k - .5) * .3, xerr=[s.pct - s.pct_lo, s.pct_hi - s.pct], fmt="o", color=c, ms=5, elinewidth=1.2, capsize=0, label=lab)
ax[1].axvline(0, color="#555", lw=.8); ax[1].set_yticks(yy, list(KEY.values())); ax[1].grid(axis="y", visible=False); ax[1].grid(axis="x", visible=True)
ax[1].set_xlabel("Adjusted % difference in spending (95% bootstrap CI)"); ax[1].set_title("Median vs mean: do the gaps change?"); ax[1].legend(loc="lower right")
fig.tight_layout(); style.save(fig, "loss_lad_vs_ols", "Source: MEPS 2022, adults 18+. Survey-weighted |y − x′θ|^a regressions of log(1 + total $) on demographics, SES, access and health.")
