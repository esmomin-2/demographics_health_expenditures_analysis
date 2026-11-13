"""14 — Interaction analysis: does the race/ethnicity gap in spending change across income, insurance, age and sex?
Survey-weighted WLS of log(1+total $) [and log OOP] with race x moderator interaction, adjusting for age, sex, SES, region,
insurance, usual source of care and health. For each moderator level: adjusted % gap vs NH White = predictive-margin contrast
(SE by delta method on the linear predictor, clustered strata x PSU). Joint Wald test of all interaction terms.
Plus ML check: XGBoost SHAP interaction values (top feature pairs)."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.formula.api as smf
from patsy import build_design_matrices
from margins import HEALTH, NEED
d = pd.read_csv("data/meps2022_adults.csv")
d["age_grp"] = pd.cut(d.age, [17, 34, 49, 64, 120], labels=["18-34", "35-49", "50-64", "65+"]).astype(str)
d["inc_grp"] = d.pov_cat.map({"Poor (<100% FPL)": "<200% FPL", "Near poor": "<200% FPL", "Low income": "<200% FPL",
                             "Middle income": "200–399% FPL", "High income": "≥400% FPL"})
d["sex"] = np.where(d.female == 1, "Female", "Male")
RACES = ["NH Black", "Hispanic", "NH Asian"]
d = d[d.race.isin(RACES + ["NH White"])].dropna(subset=NEED + ["educ_yrs", "employed", "married", "has_usual_source"]).copy()
d["grp"] = d.strata * 100 + d.psu
BASE = ["C(age_grp)", "C(sex)", "educ_yrs", "C(pov_cat)", "employed", "married", "fam_size", "C(region)", "C(insurance)", "has_usual_source", HEALTH]
MODS = {"inc_grp": ("Family income", ["<200% FPL", "200–399% FPL", "≥400% FPL"]), "insurance": ("Insurance", ["Any private", "Public only", "Uninsured"]),
        "age_grp": ("Age group", ["18-34", "35-49", "50-64", "65+"]), "sex": ("Sex", ["Female", "Male"])}
rows, tests = [], []
for y in ["log_total_exp", "log_oop_exp"]:
    for mod, (mlab, lv) in MODS.items():
        ctrl = " + ".join(c for c in BASE if not (mod == "inc_grp" and "pov_cat" in c))
        f = f"{y} ~ C(race, Treatment('NH White')) * C({mod}) + {ctrl}"
        m = smf.wls(f, d, weights=d.weight).fit(cov_type="cluster", cov_kwds={"groups": d.grp}); V = m.cov_params().values; di = m.model.data.design_info
        names = [n for n in m.params.index if "C(race" in n and ":" in n]
        Rm = np.zeros((len(names), len(m.params))); [Rm.__setitem__((i, list(m.params.index).index(n)), 1) for i, n in enumerate(names)]
        wt = m.wald_test(Rm, scalar=True); tests.append(dict(outcome=y, moderator=mlab, chi2=float(wt.statistic), df=len(names), p=float(wt.pvalue)))
        for l in lv:
            sub = d[d[mod] == l]; ws = sub.weight.values / sub.weight.sum()
            Xw = np.asarray(build_design_matrices([di], sub.assign(race="NH White"))[0])
            for r in RACES:
                L = ws @ (np.asarray(build_design_matrices([di], sub.assign(race=r))[0]) - Xw)
                b, se = float(L @ m.params.values), float(np.sqrt(L @ V @ L))
                rows.append(dict(outcome=y, moderator=mlab, level=l, race=r, coef=b, se=se, pct=100 * (np.exp(b) - 1),
                                 lo=100 * (np.exp(b - 1.96 * se) - 1), hi=100 * (np.exp(b + 1.96 * se) - 1)))
G = pd.DataFrame(rows); W = pd.DataFrame(tests)
G.round(4).to_csv("output/tables/interactions_race_gaps.csv", index=False); W.round(4).to_csv("output/tables/interactions_wald.csv", index=False)
print(W.round(4).to_string())

# no-interaction benchmark (pooled gap)
pool = smf.wls(f"log_total_exp ~ C(race, Treatment('NH White')) + {' + '.join(BASE)}", d, weights=d.weight).fit(cov_type="cluster", cov_kwds={"groups": d.grp})
PG = {r: 100 * (np.exp(pool.params[f"C(race, Treatment('NH White'))[T.{r}]"]) - 1) for r in RACES}; print("pooled", PG)

CR = {"NH Black": style.RUST, "Hispanic": style.TEAL, "NH Asian": style.GOLD}
fig, axs = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True)
for ax, (mod, (mlab, lv)) in zip(axs.flat, MODS.items()):
    g = G[(G.outcome == "log_total_exp") & (G.moderator == mlab)]
    for k, r in enumerate(RACES):
        s = g[g.race == r].set_index("level").loc[lv]; yy = np.arange(len(lv))[::-1] + (1 - k) * .22
        ax.errorbar(s.pct, yy, xerr=[s.pct - s.lo, s.hi - s.pct], fmt="o", ms=5.5, color=CR[r], elinewidth=1.3, capsize=0, label=f"{r} vs NH White")
        ax.axvline(PG[r], color=CR[r], lw=.9, ls=":", alpha=.9)
    ax.axvline(0, color="#555", lw=.8); ax.set_xlim(-85, 65); ax.set_yticks(np.arange(len(lv))[::-1], lv); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    p = W[(W.outcome == "log_total_exp") & (W.moderator == mlab)].p.iloc[0]
    ax.set_title(f"By {mlab.lower()}   (interaction p {'< 0.001' if p < .001 else f'= {p:.3f}'})")
for a in axs[1]: a.set_xlabel("Adjusted % difference in total spending vs NH White (95% CI)")
h, l = axs[0, 0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0, 1, .955))
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.0), frameon=False)
style.save(fig, "interactions_race", "Source: MEPS HC-243 (2022), adults 18+. Survey-weighted WLS of log(1 + total $) with race × moderator interaction, adjusting for age, sex, education, income, employment, marital status, "
           "family size, region, insurance, usual source of care and health; dotted lines = pooled (no-interaction) gap; CIs beyond +65% are truncated. p = joint Wald test of interaction terms.")

# ---------- ML check: XGBoost SHAP interaction values (top pairs)
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
dd = pd.read_csv("data/meps2022_adults.csv")
NUM = ["age", "female", "educ_yrs", "fam_income", "fam_size", "health_poor", "mental_poor", "married", "employed", "has_usual_source",
       "diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer", "n_chronic"]
X = pd.get_dummies(dd[NUM + ["race", "pov_cat", "insurance", "region"]], columns=["race", "pov_cat", "insurance", "region"], dtype=float)
X.columns = [c.replace("<", "lt") for c in X.columns]
tr, te = train_test_split(np.arange(len(dd)), test_size=.2, random_state=42); X = X.fillna(X.iloc[tr].median())
P = pd.read_csv("output/tables/s_xgb_best_params.csv").set_index("param").value.to_dict(); P = {k: (int(v) if k in ("n_estimators", "max_depth") else v) for k, v in P.items()}
mx = XGBRegressor(random_state=42, n_jobs=4, tree_method="hist", **P).fit(X.iloc[tr], dd.log_total_exp.values[tr], sample_weight=dd.weight.values[tr])
Xs = X.iloc[te].sample(1500, random_state=42); I = shap.TreeExplainer(mx).shap_interaction_values(Xs)
M = np.abs(I).mean(0); np.fill_diagonal(M, 0); cols = list(X.columns); pairs = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)): pairs.append((cols[i], cols[j], 2 * M[i, j]))
PI = pd.DataFrame(pairs, columns=["a", "b", "mean_abs_interaction"]).sort_values("mean_abs_interaction", ascending=False)
PI.head(25).round(4).to_csv("output/tables/interactions_shap_pairs.csv", index=False); print(PI.head(12).round(3).to_string())
main = np.abs(I.sum(2)).mean(0).sum(); print("share of SHAP magnitude from pairwise interactions:", round(PI.mean_abs_interaction.sum() / (PI.mean_abs_interaction.sum() + np.abs(np.diagonal(I, axis1=1, axis2=2)).mean(0).sum()), 3))
NICE = lambda c: (c.replace("n_chronic", "No. chronic").replace("has_usual_source", "Usual source").replace("fam_size", "Family size").replace("educ_yrs", "Education")
                  .replace("health_poor", "Poor health").replace("mental_poor", "Poor mental health").replace("insurance_", "Ins: ").replace("race_", "Race: ")
                  .replace("pov_cat_", "Income: ").replace("region_", "Region: ").replace("female", "Female").replace("age", "Age").replace("fam_income", "Family income").replace("married", "Married"))
top = PI.head(12)[::-1]
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.barh(range(len(top)), top.mean_abs_interaction, color=style.PLUM, height=.6)
ax.set_yticks(range(len(top)), [f"{NICE(a)} × {NICE(b)}" for a, b in zip(top.a, top.b)]); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
ax.set_xlabel("Mean |SHAP interaction| (log points)"); ax.set_title("Strongest feature interactions learned by XGBoost")
fig.tight_layout(); style.save(fig, "interactions_shap", "Source: MEPS HC-243 (2022), adults 18+. CV-tuned XGBoost on log(1 + total $); TreeSHAP interaction values on 1,500 test adults (pair value counted in both directions).")
