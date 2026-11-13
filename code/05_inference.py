"""
05_inference.py -- Statistical inference on demographic associations with spending (MEPS 2022 adults).

Part 1: Survey-weighted OLS (WLS, cluster-robust by PSU-within-strata), Models 1-3 + two-part model.
Part 2: Double/Debiased ML, partially linear model (mirrors course notes, Sep23/Sep28/Sep30):
          Y_i = theta0 D_i + g0(X_i) + U_i,   E[U|D,X]=0
          nuisance eta0 = (h0, d0),  h0(X)=E[Y|X],  d0(X)=E[D|X],  V_i = D_i - d0(X_i)
          moment  m(W;theta,eta) = {D - d(X)}[Y - h(X) - theta{D - d(X)}]   (Neyman-orthogonal)
        K=5-fold cross-fitting (nuisances learned off-fold), theta = residual-on-residual,
        SE from the orthogonal score: J = -E[V^2], var = E[psi^2]/(J^2 n).
        Repeated R=3 times over different splits; median aggregation (Chernozhukov et al. 2018):
          theta_med = median(theta_r), se_med = sqrt(median(se_r^2 + (theta_r - theta_med)^2)).
Part 3: AIPW / interactive regression model ATE of Uninsured on total_exp ($), propensity trimmed [0.01,0.99].
        (Extension beyond the course notes, which cover only the PLM.)
Part 4: Heterogeneity of the uninsured coefficient by age group (<65) and poverty.

All estimates are CONDITIONAL ASSOCIATIONS; causal only under unconfoundedness (E[U|D,X]=0).
DML / AIPW are unweighted (sample, not population, estimands); OLS is survey-weighted.
"""
import os, sys, warnings
sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB, FIG = os.path.join(ROOT, "output/tables"), os.path.join(ROOT, "output/figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True); os.chdir(ROOT)
SRC_DML = "Source: MEPS 2022 HC-243; adults 18+. DML unweighted (5-fold cross-fitting, 3 repeats, median); OLS survey-weighted."
SEED, K, R = 42, 5, 2

# ------------------------------------------------------------------ data
df = pd.read_csv(os.path.join(ROOT, "data/meps2022_adults.csv"))
health = ["health_poor", "mental_poor", "diabetes", "hypertension", "heart_disease",
          "asthma", "arthritis", "cancer"]
need = ["age", "female", "race", "educ_yrs", "pov_cat", "fam_size", "married", "region",
        "insurance"] + health
df = df.dropna(subset=need).reset_index(drop=True)
df["age2"] = df["age"] ** 2 / 100
df["cluster"] = df["strata"] * 100 + df["psu"]
df["log_fam_income"] = np.log1p(df["fam_income"].clip(lower=0))
for r, n in [("NH Black", "black"), ("Hispanic", "hispanic"), ("NH Asian", "asian"),
             ("NH Other/Multi", "other_race")]:
    df[n] = (df["race"] == r).astype(float)
df["poor_nearpoor"] = df["pov_cat"].isin(["Poor (<100% FPL)", "Near poor"]).astype(float)
df["pov_low"] = (df["pov_cat"] == "Low income").astype(float)
df["pov_mid"] = (df["pov_cat"] == "Middle income").astype(float)
df["uninsured"] = (df["insurance"] == "Uninsured").astype(float)
df["public_only"] = (df["insurance"] == "Public only").astype(float)
for g in ["Northeast", "Midwest", "West"]:
    df["reg_" + g] = (df["region"] == g).astype(float)
print(f"Analysis sample: {len(df):,} adults (complete cases)")

DEMO = ["age", "age2", "female", "black", "hispanic", "asian", "other_race", "educ_yrs",
        "poor_nearpoor", "pov_low", "pov_mid", "fam_size", "married",
        "reg_Northeast", "reg_Midwest", "reg_West"]          # ref: NH White, high income, South
INS = ["uninsured", "public_only"]                          # ref: any private
HLTH = health
OUTS = ["log_total_exp", "log_oop_exp", "any_exp"]

# ------------------------------------------------------------------ Part 1: survey-weighted OLS
def wls(y, X, data, logit=False):
    Xc = sm.add_constant(data[X])
    if logit:
        m = sm.GLM(data[y], Xc, family=sm.families.Binomial(), freq_weights=None,
                   var_weights=data["weight"] / data["weight"].mean())
    else:
        m = sm.WLS(data[y], Xc, weights=data["weight"])
    return m.fit(cov_type="cluster", cov_kwds={"groups": data["cluster"]})

rows = []
specs = {"M1": DEMO, "M2": DEMO + INS, "M3": DEMO + INS + HLTH}
fits = {}
for y in ["log_total_exp", "log_oop_exp"]:
    for mname, X in specs.items():
        f = wls(y, X, df); fits[(y, mname)] = f
        for v in X:
            rows.append(dict(outcome=y, model=mname, var=v, coef=f.params[v], se=f.bse[v],
                             ci_lo=f.conf_int().loc[v, 0], ci_hi=f.conf_int().loc[v, 1],
                             p=f.pvalues[v], pct=100 * (np.exp(f.params[v]) - 1),
                             n=int(f.nobs), r2=f.rsquared))
# Two-part model (Model 3 covariates)
X3 = specs["M3"]
f1 = wls("any_exp", X3, df, logit=True)
pos = df[df["total_exp"] > 0]
f2 = wls("log_total_exp", X3, pos)
for v in X3:
    rows.append(dict(outcome="any_exp", model="TPM1_logit", var=v, coef=f1.params[v], se=f1.bse[v],
                     ci_lo=f1.conf_int().loc[v, 0], ci_hi=f1.conf_int().loc[v, 1], p=f1.pvalues[v],
                     pct=100 * (np.exp(f1.params[v]) - 1), n=int(f1.nobs), r2=np.nan))  # pct = odds-ratio - 1
    rows.append(dict(outcome="log_total_exp|>0", model="TPM2_ols", var=v, coef=f2.params[v], se=f2.bse[v],
                     ci_lo=f2.conf_int().loc[v, 0], ci_hi=f2.conf_int().loc[v, 1], p=f2.pvalues[v],
                     pct=100 * (np.exp(f2.params[v]) - 1), n=int(f2.nobs), r2=f2.rsquared))
ols = pd.DataFrame(rows)
ols.to_csv(os.path.join(TAB, "i_ols_models.csv"), index=False)
print("\nOLS (log_total_exp), key coefs:")
print(ols[(ols.outcome == "log_total_exp") & ols["var"].isin(["black", "hispanic", "asian", "poor_nearpoor",
      "uninsured", "public_only", "female"])].pivot(index="var", columns="model", values="coef").round(3))

# coefficient plot
keyv = ["female", "black", "hispanic", "asian", "other_race", "poor_nearpoor", "pov_low", "pov_mid",
        "married", "uninsured", "public_only"]
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
cols = {"M1": style.NAVY, "M2": style.GOLD, "M3": style.TEAL}
for ax, y in zip(axes, ["log_total_exp", "log_oop_exp"]):
    for j, m in enumerate(["M1", "M2", "M3"]):
        s = ols[(ols.outcome == y) & (ols.model == m)].set_index("var").reindex(keyv)
        ypos = np.arange(len(keyv)) + (j - 1) * 0.25
        ax.errorbar(s.coef, ypos, xerr=[s.coef - s.ci_lo, s.ci_hi - s.coef], fmt="o", ms=4,
                    color=cols[m], label=m, capsize=2)
    ax.axvline(0, color=style.GRAY, lw=0.6); ax.set_title({"log_total_exp": "Total spending", "log_oop_exp": "Out-of-pocket spending"}[y])
    ax.set_xlabel("Coefficient (log points; 0.1 ≈ 10.5% change), 95% CI")
axes[0].set_yticks(range(len(keyv))); axes[0].set_yticklabels(keyv); axes[0].invert_yaxis()
axes[0].legend(title="Model", loc="lower left")
fig.suptitle("Weighted OLS associations across nested models", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); style.save(fig, "i_ols_coefplot", source="Source: MEPS 2022 HC-243; adults 18+, survey-weighted WLS, SEs clustered by PSU-in-strata. M1 demographics; M2 +insurance; M3 +health.")

# ------------------------------------------------------------------ Part 2: DML PLM
def learner(name, binary_target=False):
    # Regressors for both h and d (d0 = E[D|X] = P(D=1|X) for binary D -> regression on 0/1 is valid)
    if name == "LASSO":
        return make_pipeline(StandardScaler(), LassoCV(cv=3, n_alphas=15, random_state=SEED, max_iter=2000, tol=1e-3, n_jobs=-1))
    if name == "RF":
        return RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20,
                                     max_features=0.5, n_jobs=-1, random_state=SEED)
    if name == "XGB":
        return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8, tree_method="hist",
                            colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbosity=0)

def lasso_X(X):
    """LASSO gets a richer dictionary: raw X + squares of continuous vars + pairwise interactions with key vars."""
    X = X.copy()
    for c in ["age", "educ_yrs", "fam_size", "health_poor", "mental_poor"]:
        if c in X: X[c + "_sq"] = X[c] ** 2
    base = list(X.columns)
    for a in ["age", "female"]:
        if a in base:
            for b in base:
                if b != a and not b.endswith("_sq"): X[f"{a}*{b}"] = X[a] * X[b]
    return X

def dml_plm(data, y, d, X, lname, K=K, R=R, seed=SEED):
    Y = data[y].values.astype(float); D = data[d].values.astype(float)
    XX = lasso_X(data[X]) if lname == "LASSO" else data[X]
    XX = XX.values.astype(float); n = len(Y)
    thetas, ses = [], []
    for r in range(R):
        hhat, dhat = np.zeros(n), np.zeros(n)
        for tr, te in KFold(K, shuffle=True, random_state=seed + r).split(XX):
            hhat[te] = learner(lname).fit(XX[tr], Y[tr]).predict(XX[te])   # h(X) ~ E[Y|X]
            dhat[te] = learner(lname).fit(XX[tr], D[tr]).predict(XX[te])   # d(X) ~ E[D|X]
        Vt, Ut = D - dhat, Y - hhat
        th = np.sum(Vt * Ut) / np.sum(Vt * Vt)                             # residual-on-residual
        psi = Vt * (Ut - th * Vt)                                          # orthogonal score
        J = np.mean(Vt * Vt)
        se = np.sqrt(np.mean(psi ** 2) / J ** 2 / n)
        thetas.append(th); ses.append(se)
    thetas, ses = np.array(thetas), np.array(ses)
    tm = np.median(thetas)
    sm_ = np.sqrt(np.median(ses ** 2 + (thetas - tm) ** 2))
    return tm, sm_

from scipy.stats import norm
TREAT = {"black": "NH Black", "hispanic": "Hispanic", "asian": "NH Asian",
         "poor_nearpoor": "Poor/near-poor (<125% FPL)", "uninsured": "Uninsured",
         "public_only": "Public-only insurance", "female": "Female",
         "log_fam_income": "log family income"}
XALL = DEMO + INS + HLTH
res = []
for d, lab in TREAT.items():
    if d == "log_fam_income":
        X = [v for v in XALL if v not in ["poor_nearpoor", "pov_low", "pov_mid"]]  # poverty is a function of income
    else:
        X = [v for v in XALL if v != d]
    for y in OUTS:
        # OLS Model-3 analogue (weighted) for comparison, same X
        f = wls(y, [d] + X, df)
        for lname in ["LASSO", "RF", "XGB"]:
            th, se = dml_plm(df, y, d, X, lname)
            res.append(dict(treatment=d, label=lab, outcome=y, learner=lname, theta=th, se=se,
                            ci_lo=th - 1.96 * se, ci_hi=th + 1.96 * se, p=2 * (1 - norm.cdf(abs(th / se))),
                            ols_m3=f.params[d], ols_m3_se=f.bse[d], n=len(df), K=K, R=R))
            print(f"{lab:28s} {y:14s} {lname:5s} theta={th:7.3f} se={se:.3f} | OLS-M3={f.params[d]:7.3f}")
dml = pd.DataFrame(res)
dml.to_csv(os.path.join(TAB, "i_dml_results.csv"), index=False)

# forest plot
fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
labs = list(TREAT.values()); mk = {"LASSO": ("o", style.NAVY), "RF": ("s", style.TEAL), "XGB": ("D", style.PLUM)}
for ax, y in zip(axes, OUTS):
    for j, l in enumerate(["LASSO", "RF", "XGB"]):
        s = dml[(dml.outcome == y) & (dml.learner == l)].set_index("label").reindex(labs)
        yp = np.arange(len(labs)) + (j - 1.5) * 0.18
        ax.errorbar(s.theta, yp, xerr=1.96 * s.se, fmt=mk[l][0], color=mk[l][1], ms=4, capsize=2, label=f"DML-{l}")
    s = dml[(dml.outcome == y) & (dml.learner == "LASSO")].set_index("label").reindex(labs)
    ax.errorbar(s.ols_m3, np.arange(len(labs)) + 0.27, xerr=1.96 * s.ols_m3_se, fmt="o", color=style.RUST, ms=4,
                capsize=2, label="Weighted OLS M3")
    ax.axvline(0, color=style.GRAY, lw=0.6)
    ax.set_title({"log_total_exp": "Log total spending", "log_oop_exp": "Log out-of-pocket", "any_exp": "Any spending"}[y])
    ax.set_xlabel("θ, pp (any spending ×100)" if y == "any_exp" else "θ, log points (95% CI)")
axes[0].set_yticks(range(len(labs))); axes[0].set_yticklabels(labs); axes[0].invert_yaxis()
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("DML vs OLS conditional associations", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); style.save(fig, "i_dml_forest", source=SRC_DML + " Associations, not causal effects.")

# ------------------------------------------------------------------ Part 3: AIPW (IRM) ATE, Uninsured -> total_exp ($)
def aipw(data, y, d, X, seed=SEED, trim=(0.01, 0.99)):
    Y = data[y].values.astype(float); D = data[d].values.astype(int); XX = data[X].values.astype(float)
    n = len(Y); thetas, ses = [], []
    for r in range(R):
        m1, m0, p = np.zeros(n), np.zeros(n), np.zeros(n)
        for tr, te in KFold(K, shuffle=True, random_state=seed + r).split(XX):
            reg = lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                       n_jobs=-1, random_state=SEED, verbosity=0)
            t1, t0 = tr[D[tr] == 1], tr[D[tr] == 0]
            m1[te] = reg().fit(XX[t1], Y[t1]).predict(XX[te])
            m0[te] = reg().fit(XX[t0], Y[t0]).predict(XX[te])
            p[te] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, n_jobs=-1,
                                  random_state=SEED, verbosity=0).fit(XX[tr], D[tr]).predict_proba(XX[te])[:, 1]
        p = np.clip(p, *trim)
        phi = m1 - m0 + D * (Y - m1) / p - (1 - D) * (Y - m0) / (1 - p)
        thetas.append(phi.mean()); ses.append(phi.std(ddof=1) / np.sqrt(n))
    thetas, ses = np.array(thetas), np.array(ses); tm = np.median(thetas)
    return tm, np.sqrt(np.median(ses ** 2 + (thetas - tm) ** 2))

Xu = [v for v in XALL if v not in INS]   # compare uninsured vs insured (any coverage)
aipw_rows = []
for lab, sub in [("All adults", df), ("Age 18-64", df[df.age < 65])]:
    th, se = aipw(sub.reset_index(drop=True), "total_exp", "uninsured", Xu)
    raw = sub.loc[sub.uninsured == 1, "total_exp"].mean() - sub.loc[sub.uninsured == 0, "total_exp"].mean()
    aipw_rows.append(dict(sample=lab, estimand="ATE uninsured vs insured, total_exp ($)", ate=th, se=se,
                          ci_lo=th - 1.96 * se, ci_hi=th + 1.96 * se, p=2 * (1 - norm.cdf(abs(th / se))),
                          raw_diff=raw, n=len(sub), n_uninsured=int(sub.uninsured.sum())))
    print(f"AIPW {lab}: ATE={th:,.0f} (se {se:,.0f}); raw diff {raw:,.0f}")
pd.DataFrame(aipw_rows).to_csv(os.path.join(TAB, "i_aipw_uninsured.csv"), index=False)

# ------------------------------------------------------------------ Part 4: heterogeneity (uninsured, <65), DML-XGB
het = []
u65 = df[df.age < 65]
groups = [("Age 18-34", u65[u65.age < 35]), ("Age 35-49", u65[(u65.age >= 35) & (u65.age < 50)]),
          ("Age 50-64", u65[u65.age >= 50]), ("<125% FPL (18-64)", u65[u65.poor_nearpoor == 1]),
          (">=125% FPL (18-64)", u65[u65.poor_nearpoor == 0]), ("All 18-64", u65)]
Xh = [v for v in XALL if v != "uninsured"]
for lab, sub in groups:
    sub = sub.reset_index(drop=True)
    for y in ["log_total_exp", "log_oop_exp", "any_exp"]:
        th, se = dml_plm(sub, y, "uninsured", Xh, "XGB")
        het.append(dict(group=lab, outcome=y, theta=th, se=se, ci_lo=th - 1.96 * se, ci_hi=th + 1.96 * se,
                        p=2 * (1 - norm.cdf(abs(th / se))), n=len(sub), n_uninsured=int(sub.uninsured.sum())))
        print(f"HET {lab:20s} {y:14s} theta={th:6.3f} se={se:.3f}")
het = pd.DataFrame(het); het.to_csv(os.path.join(TAB, "i_dml_heterogeneity.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
s = het[het.outcome == "log_total_exp"]
ax.errorbar(s.theta, range(len(s)), xerr=1.96 * s.se, fmt="o", capsize=3, color=style.PLUM)
ax.set_yticks(range(len(s))); ax.set_yticklabels(s.group); ax.invert_yaxis(); ax.axvline(0, color=style.GRAY, lw=.6)
ax.set_xlabel("DML-XGB θ: uninsured, log points of total spending (95% CI)")
ax.set_title("Uninsured association by subgroup, ages 18-64")
fig.tight_layout(); style.save(fig, "i_dml_heterogeneity", source=SRC_DML)
print("Done.")
