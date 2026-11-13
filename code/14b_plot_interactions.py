"""14b — Re-plot Figure 7 (race-gap interactions) from saved tables; CIs that run past the axis are cropped with arrowheads."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt
G = pd.read_csv("output/tables/interactions_race_gaps.csv"); W = pd.read_csv("output/tables/interactions_wald.csv")
PG = {"NH Black": -52.8, "Hispanic": -53.2, "NH Asian": -32.1}   # pooled (no-interaction) gaps from 14_interactions.py
RACES = list(PG); CR = {"NH Black": style.RUST, "Hispanic": style.TEAL, "NH Asian": style.GOLD}
MODS = [("Age group", ["18-34", "35-49", "50-64", "65+"]), ("Sex", ["Female", "Male"])]
XL, XR = -85, 30
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, gridspec_kw=dict(width_ratios=[1.25, 1]))
for ax, (mlab, lv) in zip(axs, MODS):
    g = G[(G.outcome == "log_total_exp") & (G.moderator == mlab)]
    for k, r in enumerate(RACES):
        s = g[g.race == r].set_index("level").loc[lv]; yy = np.arange(len(lv))[::-1] + (1 - k) * .22
        lo, hi = s.lo.clip(XL + 1, XR - 1), s.hi.clip(XL + 1, XR - 1)
        ax.errorbar(s.pct, yy, xerr=[s.pct - lo, hi - s.pct], fmt="o", ms=5.5, color=CR[r], elinewidth=1.3, capsize=0, label=f"{r} vs NH White")
        for y_, h_, l_ in zip(yy, s.hi, s.lo):
            if h_ > XR - 1: ax.plot(XR - 1, y_, ">", color=CR[r], ms=6, clip_on=False)
            if l_ < XL + 1: ax.plot(XL + 1, y_, "<", color=CR[r], ms=6, clip_on=False)
        ax.axvline(PG[r], color=CR[r], lw=.9, ls=":")
    ax.axvline(0, color="#555", lw=.8); ax.set_xlim(XL, XR); ax.set_yticks(np.arange(len(lv))[::-1], lv)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    p = W[(W.outcome == "log_total_exp") & (W.moderator == mlab)].p.iloc[0]
    ax.set_title(f"By {mlab.lower()}   (interaction p {'< 0.001' if p < .001 else f'= {p:.3f}'})")
for a in axs: a.set_xlabel("Adjusted % difference in total spending vs NH White (95% CI)")
h, l = axs[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0, 1, .9))
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.0), frameon=False)
style.save(fig, "interactions_race", "Source: MEPS HC-243 (2022), adults 18+. Survey-weighted WLS of log(1 + total $) with race × moderator interaction, adjusting for age, sex, education, income, employment, marital status, "
           "family size, region, insurance, usual source of care and health. Dotted lines = pooled (no-interaction) gap; arrows = CI extends beyond axis. p = joint Wald test. Income and insurance interactions were not significant (p = 0.20, 0.29).")
