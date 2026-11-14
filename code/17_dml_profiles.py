"""17 — DML on the K-means profiles: is each profile's spending gap vs privately insured White adults explained by health?
Profiles are rebuilt exactly as in 03_unsupervised.py (demographics + SES incl. access; no health variables).
For each profile k vs the reference profile: partially linear model on the pooled pair,
    Y = θ·1[profile k] + g(health) + U,
with health (self-rated physical/mental health + six conditions) adjusted for FLEXIBLY by ML nuisance learners,
5-fold cross-fitting × 3 repeats, median aggregation, orthogonal-score SEs.
Compare θ with the raw gap (no adjustment) and with linear OLS health adjustment.
Conditional associations, not causal effects. DML is unweighted (sample estimand)."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib, warnings; matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt, statsmodels.api as sm
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

RS, K, R = 42, 5, 3
d = pd.read_csv("data/meps2022_adults.csv")
# ---- rebuild profiles exactly as 03_unsupervised.py
num = ["age", "female", "educ_yrs", "fam_size", "married", "employed", "has_usual_source"]
cat = ["race", "pov_cat", "insurance", "region"]
X = pd.concat([d[num].fillna(d[num].median()), pd.get_dummies(d[cat], prefix=cat, prefix_sep=": ", dtype=float)], axis=1)
Z = StandardScaler().fit_transform(X)
lab = KMeans(6, n_init=20, random_state=RS).fit(Z).labels_
NAMES = {0: "Asian, higher-income", 1: "Privately insured White adults", 2: "Privately insured Black & Hispanic workers",
         3: "Publicly insured, older & lower-income", 4: "Near-poor adults", 5: "Uninsured adults"}
d["profile"] = pd.Series(lab).map(NAMES)
print(d.groupby("profile").weight.sum().div(d.weight.sum()).round(3))   # should match 03 shares (0.43, 0.21, ...)
REF = "Privately insured White adults"

HEALTH = ["health_poor", "mental_poor", "diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer"]
d = d.dropna(subset=HEALTH).reset_index(drop=True)
d["clu"] = d.strata * 100 + d.psu

def learner(n):
    if n == "LASSO": return make_pipeline(StandardScaler(), LassoCV(cv=3, alphas=20, max_iter=3000, n_jobs=-1, random_state=RS))
    if n == "RF": return RandomForestRegressor(200, max_depth=8, min_samples_leaf=20, max_features=.5, n_jobs=-1, random_state=RS)
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, tree_method="hist",
                        n_jobs=-1, random_state=RS, verbosity=0)
def expand(H):  # richer dictionary for LASSO: squares + all pairwise interactions of health vars
    H = H.copy(); b = list(H.columns)
    for c in ["health_poor", "mental_poor"]: H[c + "_sq"] = H[c] ** 2
    for i, a in enumerate(b):
        for c in b[i + 1:]: H[f"{a}*{c}"] = H[a] * H[c]
    return H
def dml(sub, y, ln):
    Y = sub[y].values.astype(float); D = (sub.profile != REF).values.astype(float)
    XX = (expand(sub[HEALTH]) if ln == "LASSO" else sub[HEALTH]).values.astype(float); n = len(Y); th_, se_ = [], []
    for r in range(R):
        h, m = np.zeros(n), np.zeros(n)
        for tr, te in KFold(K, shuffle=True, random_state=RS + r).split(XX):
            h[te] = learner(ln).fit(XX[tr], Y[tr]).predict(XX[te]); m[te] = learner(ln).fit(XX[tr], D[tr]).predict(XX[te])
        V, U = D - m, Y - h; th = np.sum(V * U) / np.sum(V * V); psi = V * (U - th * V)
        th_.append(th); se_.append(np.sqrt(np.mean(psi ** 2) / np.mean(V * V) ** 2 / n))
    th_, se_ = np.array(th_), np.array(se_); t = np.median(th_); return t, np.sqrt(np.median(se_ ** 2 + (th_ - t) ** 2))

rows = []
for p in [v for v in NAMES.values() if v != REF]:
    sub = d[d.profile.isin([p, REF])].reset_index(drop=True); sub["D"] = (sub.profile == p).astype(float)
    for y in ["log_total_exp", "log_oop_exp", "any_exp"]:
        raw = sm.WLS(sub[y], sm.add_constant(sub[["D"]])).fit(cov_type="cluster", cov_kwds={"groups": sub.clu})
        Hd = pd.get_dummies(sub[HEALTH].astype({"health_poor": int, "mental_poor": int}), columns=["health_poor", "mental_poor"], drop_first=True, dtype=float)
        ols = sm.WLS(sub[y], sm.add_constant(pd.concat([sub[["D"]], Hd], axis=1))).fit(cov_type="cluster", cov_kwds={"groups": sub.clu})
        base = dict(profile=p, outcome=y, n=len(sub), n_profile=int(sub.D.sum()))
        rows.append({**base, "method": "Raw gap", "theta": raw.params["D"], "se": raw.bse["D"]})
        rows.append({**base, "method": "OLS + health", "theta": ols.params["D"], "se": ols.bse["D"]})
        for ln in ["LASSO", "RF", "XGB"]:
            t, se = dml(sub, y, ln); rows.append({**base, "method": f"DML-{ln}", "theta": t, "se": se})
            print(f"{p:45s} {y:14s} {ln:5s} θ={t:6.3f} se={se:.3f} | raw {raw.params['D']:6.3f} | OLS {ols.params['D']:6.3f}", flush=True)
T = pd.DataFrame(rows)
T["p"] = 2 * (1 - norm.cdf(abs(T.theta / T.se)))
lg = T.outcome != "any_exp"
T["pct"] = np.where(lg, 100 * (np.exp(T.theta) - 1), 100 * T.theta)          # % diff (log outcomes) or pp (any spending)
T["lo"] = np.where(lg, 100 * (np.exp(T.theta - 1.96 * T.se) - 1), 100 * (T.theta - 1.96 * T.se))
T["hi"] = np.where(lg, 100 * (np.exp(T.theta + 1.96 * T.se) - 1), 100 * (T.theta + 1.96 * T.se))
T.round(4).to_csv("output/tables/dml_profiles.csv", index=False)

# ---- figure: raw vs OLS-health vs DML (XGB, RF, LASSO), log total spending
M = [("Raw gap", "o", "#9AA3AE"), ("OLS + health", "s", style.GOLD), ("DML-LASSO", "^", style.NAVY), ("DML-RF", "v", style.TEAL), ("DML-XGB", "D", style.PLUM)]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
order = ["Uninsured adults", "Privately insured Black & Hispanic workers", "Asian, higher-income", "Publicly insured, older & lower-income", "Near-poor adults"]
yy = np.arange(len(order))[::-1]
for ax, y, xl in [(axs[0], "log_total_exp", "% difference in total spending vs privately insured White adults (95% CI)"),
                  (axs[1], "log_oop_exp", "% difference in out-of-pocket spending vs privately insured White adults (95% CI)")]:
    for k, (m, mk, c) in enumerate(M):
        s = T[(T.outcome == y) & (T.method == m)].set_index("profile").loc[order]
        ax.errorbar(s.pct, yy + (2 - k) * .14, xerr=[s.pct - s.lo, s.hi - s.pct], fmt=mk, color=c, ms=6, elinewidth=1.2, capsize=0, label=m)
    ax.axvline(0, color="#555", lw=.8); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True); ax.set_xlabel(xl, fontsize=9)
    ax.set_title({"log_total_exp": "Total spending", "log_oop_exp": "Out-of-pocket spending"}[y])
axs[0].set_yticks(yy, order); axs[0].legend(loc="lower right", fontsize=8)
fig.suptitle("DML on the profiles: gaps vs privately insured White adults after flexibly adjusting for health", x=.01, ha="left", fontweight="bold")
fig.tight_layout(); style.save(fig, "dml_profiles", "Source: MEPS 2022, adults 18+. Each profile pooled with the reference profile; health = self-rated physical/mental health + 6 conditions. "
                               "DML partially linear model, 5-fold cross-fitting × 3 repeats (unweighted); raw and OLS survey-weighted, SEs clustered by strata×PSU. Associations, not causal effects.")
print(T[T.outcome == "log_total_exp"].pivot(index="profile", columns="method", values="pct").round(0))
print("Done.")
