"""16 — DML extensions (ECON442 Sep 23/28/30 notes). Partially linear model Y = θD + g(X) + U, 5-fold cross-fitting, R = 3
repeats (median aggregation), residual-on-residual θ̂ with orthogonal-score SE.
(a) Heterogeneity: θ̂ estimated separately for adults <65 and 65+ (mirrors notes' 'regions differ in response').
(b) Nuisance-learner robustness: LASSO (expanded dictionary), random forest, XGBoost, neural net (MLP) — all listed in notes.
Treatments: NH Black, Hispanic, NH Asian, uninsured, female.  Outcome: log(1 + total $).  X = demographics, SES, insurance, health."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings; warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
K, R, SEED = 5, 3, 42
df = pd.read_csv("data/meps2022_adults.csv")
health = ["health_poor", "mental_poor", "diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer"]
df = df.dropna(subset=["age", "female", "race", "educ_yrs", "pov_cat", "fam_size", "married", "region", "insurance"] + health).reset_index(drop=True)
df["age2"] = df.age ** 2 / 100
for r, n in [("NH Black", "black"), ("Hispanic", "hispanic"), ("NH Asian", "asian"), ("NH Other/Multi", "other_race")]: df[n] = (df.race == r).astype(float)
df["poor_nearpoor"] = df.pov_cat.isin(["Poor (<100% FPL)", "Near poor"]).astype(float); df["pov_low"] = (df.pov_cat == "Low income").astype(float)
df["pov_mid"] = (df.pov_cat == "Middle income").astype(float); df["uninsured"] = (df.insurance == "Uninsured").astype(float)
df["public_only"] = (df.insurance == "Public only").astype(float)
for g in ["Northeast", "Midwest", "West"]: df["reg_" + g] = (df.region == g).astype(float)
XALL = ["age", "age2", "female", "black", "hispanic", "asian", "other_race", "educ_yrs", "poor_nearpoor", "pov_low", "pov_mid", "fam_size",
        "married", "reg_Northeast", "reg_Midwest", "reg_West", "uninsured", "public_only"] + health
def learner(n):
    if n == "LASSO": return make_pipeline(StandardScaler(), LassoCV(cv=3, n_alphas=20, max_iter=3000, n_jobs=-1, random_state=SEED))
    if n == "RF": return RandomForestRegressor(200, max_depth=10, min_samples_leaf=20, max_features=.5, n_jobs=-1, random_state=SEED)
    if n == "XGB": return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)
    if n == "NN": return make_pipeline(StandardScaler(), MLPRegressor((64, 32), alpha=1e-2, max_iter=300, early_stopping=True, random_state=SEED))
def expand(X):
    X = X.copy()
    for c in ["age", "educ_yrs", "fam_size", "health_poor", "mental_poor"]: X[c + "_sq"] = X[c] ** 2
    for a in ["age", "female"]:
        for b in list(X.columns):
            if b != a and not b.endswith("_sq"): X[f"{a}*{b}"] = X[a] * X[b]
    return X
def dml(data, y, d, X, ln):
    Y, D = data[y].values, data[d].values; XX = (expand(data[X]) if ln == "LASSO" else data[X]).values.astype(float); n = len(Y); th_, se_ = [], []
    for r in range(R):
        h, m = np.zeros(n), np.zeros(n)
        for tr, te in KFold(K, shuffle=True, random_state=SEED + r).split(XX):
            h[te] = learner(ln).fit(XX[tr], Y[tr]).predict(XX[te]); m[te] = learner(ln).fit(XX[tr], D[tr]).predict(XX[te])
        V, U = D - m, Y - h; th = np.sum(V * U) / np.sum(V * V); psi = V * (U - th * V)
        th_.append(th); se_.append(np.sqrt(np.mean(psi ** 2) / np.mean(V * V) ** 2 / n))
    th_, se_ = np.array(th_), np.array(se_); t = np.median(th_); return t, np.sqrt(np.median(se_ ** 2 + (th_ - t) ** 2))
TREAT = {"black": "NH Black", "hispanic": "Hispanic", "asian": "NH Asian", "uninsured": "Uninsured", "female": "Female"}
GROUPS = {"All adults": df, "Age 18-64": df[df.age < 65], "Age 65+": df[df.age >= 65]}
rows = []
for gname, g in GROUPS.items():
    for d, lab in TREAT.items():
        X = [v for v in XALL if v != d]
        for ln in (["LASSO", "RF", "XGB", "NN"] if gname == "All adults" else ["XGB"]):
            t, se = dml(g, "log_total_exp", d, X, ln)
            rows.append(dict(group=gname, treatment=lab, learner=ln, theta=t, se=se, p=2 * (1 - norm.cdf(abs(t / se))), n=len(g),
                             pct=100 * (np.exp(t) - 1), lo=100 * (np.exp(t - 1.96 * se) - 1), hi=100 * (np.exp(t + 1.96 * se) - 1)))
            print(gname, lab, ln, round(t, 3), round(se, 3), flush=True)
Rz = pd.DataFrame(rows); Rz.round(4).to_csv("output/tables/dml_extensions.csv", index=False)
# test difference <65 vs 65+
H = Rz[Rz.learner == "XGB"].pivot(index="treatment", columns="group", values=["theta", "se"])
diff = pd.DataFrame({"diff": H["theta"]["Age 65+"] - H["theta"]["Age 18-64"], "se": np.sqrt(H["se"]["Age 65+"] ** 2 + H["se"]["Age 18-64"] ** 2)})
diff["p"] = 2 * (1 - norm.cdf(abs(diff["diff"] / diff["se"]))); diff.round(4).to_csv("output/tables/dml_age_difference_test.csv"); print(diff.round(3))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
labs = list(TREAT.values()); yy = np.arange(len(labs))[::-1]
for k, (gname, c) in enumerate([("Age 18-64", style.NAVY), ("Age 65+", style.RUST)]):
    s = Rz[(Rz.group == gname) & (Rz.learner == "XGB")].set_index("treatment").loc[labs]
    ax[0].errorbar(s.pct, yy + (.5 - k) * .3, xerr=[s.pct - s.lo, s.hi - s.pct], fmt="o", color=c, ms=6, elinewidth=1.3, capsize=0, label=gname)
for i, l in enumerate(labs):
    p = diff.loc[l, "p"]; ax[0].text(ax[0].get_xlim()[1] if False else 62, yy[i], f"p={p:.3f}" if p >= .001 else "p<0.001", va="center", fontsize=8, color="#444")
ax[0].axvline(0, color="#555", lw=.8); ax[0].set_xlim(-95, 75); ax[0].set_yticks(yy, labs); ax[0].grid(axis="y", visible=False); ax[0].grid(axis="x", visible=True)
ax[0].set_xlabel("DML θ as % difference in total spending (95% CI)"); ax[0].set_title("Heterogeneity: working age vs Medicare age"); ax[0].legend(loc="lower left")
M = {"LASSO": ("o", style.NAVY), "RF": ("s", style.TEAL), "XGB": ("D", style.PLUM), "NN": ("^", style.GOLD)}
for k, (ln, (mk, c)) in enumerate(M.items()):
    s = Rz[(Rz.group == "All adults") & (Rz.learner == ln)].set_index("treatment").loc[labs]
    ax[1].errorbar(s.pct, yy + (1.5 - k) * .17, xerr=[s.pct - s.lo, s.hi - s.pct], fmt=mk, color=c, ms=5, elinewidth=1.1, capsize=0, label={"NN": "Neural net"}.get(ln, ln))
ax[1].axvline(0, color="#555", lw=.8); ax[1].set_yticks(yy, [""] * len(labs)); ax[1].grid(axis="y", visible=False); ax[1].grid(axis="x", visible=True)
ax[1].set_xlabel("DML θ as % difference in total spending (95% CI)"); ax[1].set_title("Robustness to the nuisance learner (all adults)"); ax[1].legend(loc="lower left", ncol=2)
fig.tight_layout(); style.save(fig, "dml_extensions", "Source: MEPS 2022, adults 18+. DML partially linear model, 5-fold cross-fitting × 3 repeats; conditional associations, not causal effects.")
