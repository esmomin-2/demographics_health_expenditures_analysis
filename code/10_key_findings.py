"""10 — Main Figure 10: key findings vs. literature (CAPS-style central illustration). Numbers read from output/tables."""
import os, sys, textwrap, numpy as np, pandas as pd
sys.path.insert(0, "code"); import style
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "text.parse_math": False})
NAVY, NAVY2, INK, GREY, RED, RULE = "#16365c", "#2b5d8f", "#1e2a38", "#5d6b7c", "#c0392b", "#9fb3cb"
CONF, NOVEL = "#2b5d8f", "#c0392b"
T = "output/tables/"
# ---- numbers
d = pd.read_csv("data/meps2022_adults.csv").sort_values("total_exp", ascending=False)
cw = d.weight.cumsum() / d.weight.sum(); ce = (d.weight * d.total_exp).cumsum() / (d.weight * d.total_exp).sum()
top5 = float(ce[cw <= .05].iloc[-1]); tp = d[cw <= .05]; old = np.average(tp.age >= 65, weights=tp.weight)
t1 = pd.read_csv(T + "t1_weighted_means_by_group.csv").set_index("group")
bw = t1.loc["NH Black", "total_exp"] / t1.loc["NH White", "total_exp"]
o = pd.read_csv(T + "o_race_by_income_health_adjusted.csv")
hb = lambda r: o[(o.level == "<200% FPL") & (o.race == r) & (o.outcome == "hb100")]["mean"].iloc[0]
N = pd.read_csv(T + "o_nested_race_gaps.csv"); g = lambda m, r: N[(N.outcome == "High burden (pp)") & (N.model == m) & (N.race == r)].coef.iloc[0]
prof = pd.read_csv(T + "u_cluster_profiles.csv", index_col=0)
osh = prof.loc["oop_share_income"]; old_c = [c for c in prof.columns if "Older" in c][0]; aff = [c for c in prof.columns if "affluent" in c][0]
m = pd.read_csv(T + "s_regression_metrics.csv").set_index("model"); r2o, r2b = m.loc["OLS", "R2"], m.R2.max()
tp2 = pd.read_csv(T + "s_twopart_vs_onepart_dollars.csv"); r2d = tp2[tp2.model.str.startswith("Two-part")].R2_levels.max()
G = pd.read_csv(T + "interactions_race_gaps.csv"); gg = lambda r, l: G[(G.outcome == "log_total_exp") & (G.moderator == "Age group") & (G.race == r) & (G.level == l)].pct.iloc[0]
sh = pd.read_csv(T + "s_shap_mean_abs.csv"); rank_us = list(sh.feature).index("has_usual_source") + 1

CONFIRM = [
    (f"Top 5% → {top5:.0%}", "of all adult spending", "AHRQ MEPS briefs: 48–51% (2009–2018 data)"),
    (f"{old:.0%} of top 5%", "are adults 65+", "AHRQ 2022: 40.5% of the top 5% are 65+"),
    (f"Black/White = {bw:.2f}", "ratio of mean total spending", "Dieleman et al. 2021 (JAMA); Dickman et al. 2022 (JAMA Netw Open)"),
    (f"R² {r2o:.2f} → {r2b:.2f}", "OLS → gradient boosting (log $); ~{:.0%} in $".format(r2d), "Flexible ML adds little without claims data (Buntin & Zaslavsky 2004; Rose 2016)"),
]
NEW = [
    ("Low-income White adults feel more cost strain", f"{hb('NH White'):.0f}% vs {hb('NH Black'):.0f}%",
     f"of low-income White vs Black adults pay >10% of family income out of pocket (Hispanic {hb('Hispanic'):.0f}%; adjusted for age, sex, health).",
     "Lower bills for Black and Hispanic adults likely mean they get less care, not that care is more affordable for them."),
    ("Burden-based population profiles", f"{100 * osh[old_c]:.1f}% vs {100 * osh[aff]:.1f}%",
     "of income paid out of pocket: 'older chronically ill' vs 'healthy affluent' profile (K-means, K = 7).",
     "Prior segmentation targets high-need patients only; these profiles cover all adults and link spending to financial burden."),
    ("Access predicts like illness", f"#{rank_us} of 21 predictors",
     "Usual source of care is the 2nd-strongest SHAP driver of spending, behind chronic conditions and ahead of age and income.",
     "Being connected to care, not demographics per se, separates spenders; uninsured status lowers predicted spending most (−1.8 log points)."),
    ("Race gaps shrink after 65", f"{gg('NH Black', '18-34'):.0f}% → {gg('NH Black', '65+'):.0f}%",
     f"Black vs White adjusted spending gap at ages 18–34 vs 65+ (Hispanic: {gg('Hispanic', '18-34'):.0f}% → {gg('Hispanic', '65+'):.0f}%); age × race interaction p < 0.001.",
     "Gaps persist at every income and insurance level but narrow under near-universal Medicare coverage, consistent with an access explanation."),
]
W, H = 17.0, 10.0
fig = plt.figure(figsize=(W, H), dpi=240); A = fig.add_axes([0, 0, 1, 1]); A.set_xlim(0, W); A.set_ylim(0, H); A.axis("off")
A.add_patch(Rectangle((0, H - .95), W, .95, fc=NAVY, ec="none"))
A.text(.35, H - .475, "Key Findings: What Confirms Prior Work and What This Project Adds", fontsize=19, fontweight="bold", color="white", va="center")
logo = plt.imread("assets/emory_logo_white.png"); lh = .42; lw = lh * logo.shape[1] / logo.shape[0]
la = fig.add_axes([(W - .35 - lw) / W, (H - .475 - lh / 2) / H, lw / W, lh / H]); la.imshow(logo); la.axis("off")
TOP, BOT = H - 1.55, 1.0; LX0, LX1 = .35, 6.35; RX0, RX1 = 6.75, W - .35
for x0, x1, lab, c in ((LX0, LX1, "CONSISTENT WITH THE LITERATURE", CONF), (RX0, RX1, "WHAT THIS PROJECT ADDS", NOVEL)):
    A.text(x0, TOP + .3, lab, fontsize=11.5, fontweight="bold", color=c, va="center"); A.plot([x0, x1], [TOP + .1] * 2, color=c, lw=1.6)
# left cards
ch = (TOP - BOT - .2 * 3) / 4
for i, (big, sub, lit) in enumerate(CONFIRM):
    y = TOP - (i + 1) * ch - i * .2
    A.add_patch(Rectangle((LX0, y), LX1 - LX0, ch, fc="#f3f6fa", ec=CONF, lw=1.2))
    A.add_patch(Rectangle((LX0, y), .12, ch, fc=CONF, ec="none"))
    A.text(LX0 + .35, y + ch - .42, big, fontsize=17, fontweight="bold", color=NAVY, va="center")
    A.text(LX0 + .35, y + ch - .85, sub, fontsize=10.5, color=INK, va="center")
    A.text(LX0 + .35, y + .3, "✓ " + lit, fontsize=9.2, color=GREY, va="center", style="italic")
# right cards
rh = (TOP - BOT - .2 * (len(NEW) - 1)) / len(NEW)
for i, (ttl, big, what, why) in enumerate(NEW):
    y = TOP - (i + 1) * rh - i * .2
    A.add_patch(Rectangle((RX0, y), RX1 - RX0, rh, fc="white", ec=NOVEL, lw=1.6))
    A.add_patch(Rectangle((RX0, y + rh - .46), RX1 - RX0, .46, fc=NOVEL, ec="none"))
    A.text(RX0 + .2, y + rh - .23, f"NOVEL {i + 1}  ·  {ttl}", fontsize=12, fontweight="bold", color="white", va="center")
    A.text(RX0 + .25, y + rh - .95, big, fontsize=21, fontweight="bold", color=NOVEL, va="center")
    A.text(RX0 + 4.3, y + rh - .98, textwrap.fill(what, 78), fontsize=10, color=INK, va="center", linespacing=1.3)
    A.plot([RX0 + .2, RX1 - .2], [y + .58] * 2, color=RULE, lw=.8)
    A.text(RX0 + .25, y + .32, textwrap.fill("Why it matters: " + why, 125), fontsize=10, color=NAVY, va="center", linespacing=1.3)
A.text(.35, .62, "Cross-sectional MEPS 2022 (HC-243), 17,387 adults, survey-weighted. All results are associations, not causal effects. "
       "'Novel' = not found in the reviewed literature for 2022 data; scope of a course project.", fontsize=9, color=GREY, va="center", style="italic")
A.text(.35, .3, textwrap.fill("Abbreviations: " + style._abbrevs(fig), 260), fontsize=8.5, color=GREY, va="center", linespacing=1.3)
fig.savefig("output/figures/f9_key_findings.png", facecolor="white")
print(CONFIRM, NEW, sep="\n")
