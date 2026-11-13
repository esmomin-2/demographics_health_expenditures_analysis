"""08 — Focused question: How do racial/ethnic differences in out-of-pocket (OOP) spending vary by family income and
insurance coverage?  Two outcomes: OOP dollars (who pays more) and family OOP burden as % of family income (who faces
more strain; 'high burden' = family OOP > 10% of family income).  Survey-weighted, design-based SEs (strata x PSU).
Nested models show how gaps change as insurance and health are added (adjustment changes the estimand; not causal)."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.formula.api as smf
raw = pd.read_stata("data/h243.dta", convert_categoricals=False,
                    columns=["DUPERSID", "DUID", "FAMIDYR", "TOTSLF22", "FAMINC22"])
raw["DUPERSID"] = raw.DUPERSID.astype("int64"); raw["fam"] = raw.DUID.astype(str) + "_" + raw.FAMIDYR.astype(str)
famoop = raw.groupby("fam").TOTSLF22.sum().rename("fam_oop")             # all members incl. children
d = pd.read_csv("data/meps2022_adults.csv").merge(raw[["DUPERSID", "fam"]], left_on="dupersid", right_on="DUPERSID")
d = d.merge(famoop, left_on="fam", right_index=True)
d["burden"] = d.fam_oop / d.fam_income.clip(lower=1000)                    # floor avoids division by ~0
d["high_burden"] = (d.burden > 0.10).astype(float)
d["burden_pct"] = 100 * d.burden.clip(upper=1)                               # top-code at 100% of income
d["hb100"] = 100 * d.high_burden
d["inc_grp"] = d.pov_cat.map({"Poor (<100% FPL)": "<200% FPL", "Near poor": "<200% FPL", "Low income": "<200% FPL",
                             "Middle income": "200–399% FPL", "High income": "≥400% FPL"})
d["grp"] = d.strata * 100 + d.psu
RACES = ["NH White", "NH Black", "Hispanic", "NH Asian"]
d = d[d.race.isin(RACES)].copy()

def svy(df, mask, y):
    w = df.weight.values * mask; W = w.sum(); m = np.sum(w * df[y].values) / W; z = w * (df[y].values - m) / W
    g = pd.DataFrame({"z": z, "h": df.strata, "p": df.psu}).groupby(["h", "p"]).z.sum().reset_index()
    v = sum(len(x) / (len(x) - 1) * ((x.z - x.z.mean()) ** 2).sum() for _, x in g.groupby("h") if len(x) > 1)
    return m, np.sqrt(v), int(mask.sum())

rows = []
for by in ["inc_grp", "insurance"]:
    for r in RACES:
        for lv in d[by].unique():
            mk = ((d.race == r) & (d[by] == lv)).values.astype(float)
            for y in ["oop_exp", "hb100"]:
                m, se, n = svy(d, mk, y); rows.append(dict(by=by, race=r, level=lv, outcome=y, mean=m, se=se, n=n))
T = pd.DataFrame(rows); T.round(3).to_csv("output/tables/o_race_by_income_insurance.csv", index=False)

# ---- nested survey-weighted models: race gaps (vs NH White) as controls are added
base = "C(race, Treatment('NH White'))"
SPEC = {"M1 Race only": "", "M2 + age, sex": " + age + I(age**2) + female",
        "M3 + income": " + age + I(age**2) + female + C(pov_cat)",
        "M4 + insurance": " + age + I(age**2) + female + C(pov_cat) + C(insurance)",
        "M5 + health": " + age + I(age**2) + female + C(pov_cat) + C(insurance) + health_poor + mental_poor + n_chronic"}
out = []
dd = d.dropna(subset=["health_poor", "mental_poor", "n_chronic"])
for y, lab in [("log_oop_exp", "log OOP $"), ("hb100", "High burden (pp)")]:
    for name, rhs in SPEC.items():
        f = smf.wls(f"{y} ~ {base}{rhs}", dd, weights=dd.weight).fit(cov_type="cluster", cov_kwds={"groups": dd.grp})
        for r in RACES[1:]:
            k = f"{base}[T.{r}]"; out.append(dict(outcome=lab, model=name, race=r, coef=f.params[k], se=f.bse[k], p=f.pvalues[k]))
N = pd.DataFrame(out); N.round(4).to_csv("output/tables/o_nested_race_gaps.csv", index=False)
print(N.round(3).to_string())

# ---- interaction test: does the Black/Hispanic/Asian gap differ by income group / insurance? (joint Wald)
for by in ["inc_grp", "insurance"]:
    for y in ["log_oop_exp", "hb100"]:
        f = smf.wls(f"{y} ~ {base}*C({by}) + age + I(age**2) + female", dd, weights=dd.weight).fit(cov_type="cluster", cov_kwds={"groups": dd.grp})
        names = [n for n in f.params.index if ":" in n]
        R = np.zeros((len(names), len(f.params))); [R.__setitem__((i, list(f.params.index).index(n)), 1) for i, n in enumerate(names)]
        wt = f.wald_test(R, scalar=True); print(f"Interaction race x {by} on {y}: chi2={float(wt.statistic):.1f}, df={len(names)}, p={float(wt.pvalue):.4f}")

# ---- benchmark vs AHRQ: share of 65+ among top 5% spenders
s = d.sort_values("total_exp", ascending=False)  # (restricted to 4 race groups; recompute on full sample below)
a = pd.read_csv("data/meps2022_adults.csv").sort_values("total_exp", ascending=False)
top = a[(a.weight.cumsum() / a.weight.sum()) <= 0.05]
print("65+ share of adults:", round(np.average(a.age >= 65, weights=a.weight), 3), " 65+ share of top-5%:", round(np.average(top.age >= 65, weights=top.weight), 3))
print("Top-5% with >=2 of our 6 chronic conditions:", round(np.average(top.n_chronic.fillna(0) >= 2, weights=top.weight), 3))

# ---- health-adjusted margins for Figure A: race within income group, controlling for age, sex and health status
sys.path.insert(0, "code"); from margins import adjusted_means, HEALTH
TA = []
for y in ["oop_exp", "hb100"]:
    a = adjusted_means(d, y, "race", "age + I(age**2) + female + " + HEALTH, by="inc_grp")
    TA.append(a.assign(outcome=y, **{"by_": "inc_grp"}))
TA = pd.concat(TA).rename(columns={"by": "level", "level": "race", "adj_mean": "mean"}); TA["by"] = "inc_grp"
TA.round(3).to_csv("output/tables/o_race_by_income_health_adjusted.csv", index=False); print(TA.round(1).to_string())
T_unadj = T; T = TA
# ---- FIGURE A: race x income, two outcomes (dot + CI)
CR = {"NH White": style.NAVY, "NH Black": style.RUST, "Hispanic": style.TEAL, "NH Asian": style.GOLD}
ORD = {"inc_grp": ["<200% FPL", "200–399% FPL", "≥400% FPL"], "insurance": ["Any private", "Public only", "Uninsured"]}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for i, y in enumerate(["oop_exp", "hb100"]):
    for j, by in enumerate(["inc_grp"]):
        ax = axs[i]; lv = ORD[by]
        for k, r in enumerate(RACES):
            s_ = T[(T.by == by) & (T.race == r) & (T.outcome == y)].set_index("level").loc[lv]
            x = np.arange(len(lv)) + (k - 1.5) * 0.17
            ax.errorbar(x, s_["mean"], yerr=1.96 * s_["se"], fmt="o", ms=6, color=CR[r], ecolor=CR[r], elinewidth=1.2, capsize=0, label=r)
        ax.set_xticks(range(len(lv)), lv); ax.grid(axis="x", visible=False)
        ax.set_ylabel("Adjusted mean OOP spending per adult ($)" if y == "oop_exp" else "Adjusted % in high-burden families")
        ax.set_title("Who pays more: out-of-pocket dollars" if y == "oop_exp" else "Who is strained: OOP > 10% of family income"); ax.set_xlabel("Family income (% of federal poverty line)")
        ax.set_ylim(bottom=0)
h, l = axs[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False); style.save(fig, "o_race_income_insurance", "Source: MEPS HC-243 (2022), adults 18+. Survey-weighted predictive margins adjusting for age, sex, self-rated physical/mental health and 6 chronic conditions; 95% CIs (strata×PSU). Burden = all family members' OOP ÷ family income.")

# ---- FIGURE B: nested-model race gaps
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
mods = ["M1 Race only", "M3 + income", "M5 + health"]; yy = np.arange(len(mods))[::-1]
for ax, lab in zip(axs, ["log OOP $", "High burden (pp)"]):
    for k, r in enumerate(RACES[1:]):
        s_ = N[(N.outcome == lab) & (N.race == r)].set_index("model").loc[mods]
        v = 100 * (np.exp(s_.coef) - 1) if lab == "log OOP $" else s_.coef
        lo = 100 * (np.exp(s_.coef - 1.96 * s_.se) - 1) if lab == "log OOP $" else s_.coef - 1.96 * s_.se
        hi = 100 * (np.exp(s_.coef + 1.96 * s_.se) - 1) if lab == "log OOP $" else s_.coef + 1.96 * s_.se
        ax.errorbar(v, yy + (k - 1) * 0.2, xerr=[v - lo, hi - v], fmt="o", ms=5, color=CR[r], elinewidth=1.2, capsize=0, label=f"{r} vs NH White")
    ax.axvline(0, color=style.GRAY, lw=.8); ax.set_yticks(yy, ["Unadjusted", "+ age, sex, income", "+ insurance, health"]); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("% difference in OOP $ (95% CI)" if lab == "log OOP $" else "Percentage-point difference (95% CI)")
    ax.set_title("OOP dollars" if lab == "log OOP $" else "High OOP burden (>10% of income)")
h, l = axs[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False); style.save(fig, "o_nested_race_gaps", "Source: MEPS HC-243 (2022), adults 18+. Survey-weighted WLS, SEs clustered by strata×PSU. Each row adds controls; adjustment changes the question, not causal.")
