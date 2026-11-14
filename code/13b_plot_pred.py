"""13b — Figure 10 replot (dumbbell): how much of the full model's predictive power do demographics + SES alone deliver?"""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib.pyplot as plt
R = pd.read_csv("output/tables/pred_ses_vs_full.csv").pivot(index="model", columns="features", values="test_R2")
A, F = "Demographics + SES only", "Full (+ health)"
R = R.sort_values(F); y = np.arange(len(R))
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.hlines(y, R[A], R[F], color="#C9CED6", lw=3, zorder=1)
ax.scatter(R[A], y, s=70, color=style.TEAL, zorder=3, label="Demographics + SES (incl. access to care)")
ax.scatter(R[F], y, s=70, color=style.NAVY, zorder=3, label="+ health status & clinical history")
for yi, (a, f) in enumerate(zip(R[A], R[F])):
    ax.text(a - .006, yi, f"{a:.2f}", ha="right", va="center", fontsize=8.5, color=style.TEAL)
    ax.text(f + .006, yi, f"{f:.2f}", ha="left", va="center", fontsize=8.5, color=style.NAVY)
    ax.text(.455, yi, f"{a / f:.0%}", ha="center", va="center", fontsize=8.5, color="#333")
ax.text(.455, len(R) - .35, "SES share\nof full R²", ha="center", va="bottom", fontsize=8, color="#555", linespacing=1.1)
ax.set_yticks(y, R.index); ax.set_xlim(.15, .48); ax.set_ylim(-.6, len(R) + .2)
ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
ax.set_xlabel("Test R² on log(1 + total spending)"); ax.set_title(f"Demographics + SES (incl. access) alone get ~{(R[A] / R[F]).median():.0%} of the way")
h, l = ax.get_legend_handles_labels(); fig.tight_layout(rect=(0, 0, 1, .92)); fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.0), frameon=False, fontsize=9); style.save(fig, "pred_ses_vs_full", "Source: MEPS 2022, adults 18+. Same 80/20 split and 5-fold CV tuning for every model; test R² survey-weighted.")
