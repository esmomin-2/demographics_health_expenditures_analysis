"""12 — What drives spending, by domain: health conditions, demographics, socioeconomic status (SES), access to care.
(a) Survey-weighted WLS of log(1+total $) on all factors jointly (SEs clustered by strata x PSU) -> adjusted % differences.
(b) Unadjusted weighted mean $ with vs without each condition.  (c) Domain importance: LMG share of R² and summed mean |SHAP|.
Outputs: drivers_conditions.png, drivers_demographics_ses.png, drivers_domains.png; tables drivers_wls.csv."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.formula.api as smf
d = pd.read_csv("data/meps2022_adults.csv")
d["age_grp"] = pd.cut(d.age, [17, 34, 49, 64, 120], labels=["18-34", "35-49", "50-64", "65+"]).astype(str)
d["educ"] = pd.cut(d.educ_yrs, [-1, 11, 12, 15, 30], labels=["<HS", "HS", "Some college", "BA+"]).astype(str)
d["fam_grp"] = pd.cut(d.fam_size, [0, 1, 2, 4, 30], labels=["1", "2", "3-4", "5+"]).astype(str)
d["grp"] = d.strata * 100 + d.psu
COND = ["diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer"]
dd = d.dropna(subset=COND + ["health_poor", "mental_poor", "employed", "has_usual_source", "educ_yrs", "married"]).copy()
dd = dd[dd.educ != "nan"]
f = ("log_total_exp ~ C(age_grp, Treatment('18-34')) + female + C(race, Treatment('NH White')) + C(region, Treatment('South'))"
     " + C(educ, Treatment('HS')) + C(pov_cat, Treatment('Middle income')) + employed + married + C(fam_grp, Treatment('1'))"
     " + C(insurance, Treatment('Any private')) + has_usual_source + C(health_poor) + C(mental_poor) + " + " + ".join(COND))
m = smf.wls(f, dd, weights=dd.weight).fit(cov_type="cluster", cov_kwds={"groups": dd.grp})
ci = m.conf_int(); T = pd.DataFrame({"coef": m.params, "lo": ci[0], "hi": ci[1], "p": m.pvalues})
T["pct"], T["pct_lo"], T["pct_hi"] = [100 * (np.exp(T[c]) - 1) for c in ["coef", "lo", "hi"]]
T.round(4).to_csv("output/tables/drivers_wls.csv"); print("R2", round(m.rsquared, 3), "n", int(m.nobs))
def row(name):
    k = [i for i in T.index if name in i]; assert len(k) == 1, (name, k); return T.loc[k[0]]

# ---------- Figure A: health-condition figure DROPPED (not a focus). Conditions stay in the model as controls only.
# ---------- Figure B: demographics, SES, access (adjusted)
DOM = [("Demographics", style.NAVY, [("Age 35-49 vs 18-34", "35-49"), ("Age 50-64 vs 18-34", "50-64"), ("Age 65+ vs 18-34", "[T.65+]"), ("Female vs male", "female"),
        ("NH Black vs NH White", "NH Black"), ("Hispanic vs NH White", "Hispanic"), ("NH Asian vs NH White", "NH Asian"),
        ("Married", "married"), ("Family of 2 vs living alone", "fam_grp, Treatment('1'))[T.2]"), ("Family of 5+ vs living alone", "fam_grp, Treatment('1'))[T.5+]")]),
       ("Socioeconomic status", style.TEAL, [("< High school vs HS", "<HS"), ("Some college vs HS", "Some college"), ("BA+ vs HS", "BA+"),
        ("Poor (<100% FPL) vs middle", "Poor (<100% FPL)"), ("Near poor vs middle", "Near poor"), ("Low income vs middle", "[T.Low income]"),
        ("High income vs middle", "High income"), ("Employed", "employed")]),
       ("SES: access to care", style.GOLD, [("Public only vs private", "Public only"), ("Uninsured vs private", "Uninsured"), ("Has usual source of care", "has_usual_source")])]
fig, ax = plt.subplots(figsize=(10, 9)); yv, labs, pos, his = [], [], 0, []
for dom, col, its in DOM:
    ax.text(-104, -pos + .75, dom.upper(), fontsize=9.5, fontweight="bold", color=col, va="center", ha="left")
    for l, key in its:
        r_ = row(key); ax.errorbar(r_.pct, -pos, xerr=[[r_.pct - r_.pct_lo], [r_.pct_hi - r_.pct]], fmt="o", color=col, ms=6, elinewidth=1.3, capsize=0)
        ax.text(max(r_.pct_hi, r_.pct) + 4, -pos, f"{r_.pct:+.0f}%", va="center", fontsize=8, color="#333")
        yv.append(-pos); labs.append(l); his.append(r_.pct_hi); pos += 1
    pos += 1.2
ax.set_yticks(yv, labs); ax.axvline(0, color=style.GRAY, lw=.8); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
ax.set_xlim(-108, max(his) + 45); ax.set_ylim(-pos + 0.6, 1.4)
ax.set_xlabel("Adjusted % difference in total spending (95% CI)"); ax.set_title("Demographic, socioeconomic and access drivers (holding health constant)")
fig.tight_layout(); style.save(fig, "drivers_demographics_ses", "Source: MEPS HC-243 (2022), adults 18+. Same joint survey-weighted WLS as the conditions figure (also adjusts for self-rated health and 6 conditions); 95% CIs clustered by strata×PSU.")

# ---------- Figure C: domain importance (LMG R² shares + SHAP)
L = pd.read_csv("output/tables/s_lmg.csv")
DMAP = {"health": "Health conditions", "insurance": "Socioeconomic status (incl. access)", "age": "Demographics", "family": "Demographics", "race": "Demographics",
        "sex": "Demographics", "education": "Socioeconomic status (incl. access)", "income/poverty": "Socioeconomic status (incl. access)", "region": "Demographics"}
L["domain"] = L.block.map(DMAP); Ld = L.groupby("domain").lmg_r2.sum()
S = pd.read_csv("output/tables/s_shap_mean_abs.csv"); S.columns = ["feature", "imp", "dir"]
SM = {**{c: "Health conditions" for c in ["n_chronic", "health_poor", "mental_poor", "diabetes", "arthritis", "hypertension", "cancer", "asthma", "heart_disease"]},
      **{c: "Demographics" for c in ["fam_size", "race", "female", "age", "married", "region"]},
      **{c: "Socioeconomic status (incl. access)" for c in ["educ_yrs", "pov_cat", "employed", "fam_income"]}, "insurance": "Socioeconomic status (incl. access)", "has_usual_source": "Socioeconomic status (incl. access)"}
S["domain"] = S.feature.map(SM); Sd = S.groupby("domain").imp.sum()
DC = {"Health conditions": style.RUST, "Demographics": style.NAVY, "Socioeconomic status (incl. access)": style.TEAL}
order = ["Health conditions", "Demographics", "Socioeconomic status (incl. access)"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); yy = np.arange(3)[::-1]
ax[0].barh(yy, [100 * Ld[o] / Ld.sum() for o in order], color=[DC[o] for o in order], height=.6)
for v, y_ in zip([100 * Ld[o] / Ld.sum() for o in order], yy): ax[0].text(v + 1, y_, f"{v:.0f}%", va="center", fontsize=9)
ax[0].set_yticks(yy, order); ax[0].set_xlabel("Share of explained variance (LMG, % of R²)"); ax[0].set_title(f"Linear model: who explains the R² of {Ld.sum():.2f}")
ax[1].barh(yy, [100 * Sd[o] / Sd.sum() for o in order], color=[DC[o] for o in order], height=.6)
for v, y_ in zip([100 * Sd[o] / Sd.sum() for o in order], yy): ax[1].text(v + 1, y_, f"{v:.0f}%", va="center", fontsize=9)
ax[1].set_yticks(yy, [""] * 3); ax[1].set_xlabel("Share of total mean |SHAP| (%)"); ax[1].set_title("XGBoost: share of prediction influence")
for a in ax: a.grid(axis="y", visible=False); a.grid(axis="x", visible=True); a.set_xlim(0, 70)
fig.tight_layout(); style.save(fig, "drivers_domains", "Source: MEPS HC-243 (2022), adults 18+. LMG = Shapley decomposition of OLS R² over feature blocks; SHAP from CV-tuned XGBoost on log(1 + total $). Demographics = age, sex, race, family, region; SES = income, poverty, education, employment, insurance, usual source of care.")
print(Ld / Ld.sum(), Sd / Sd.sum(), sep="\n")
