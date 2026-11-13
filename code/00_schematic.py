"""00 — Central-illustration-style analysis schematic (matches CAPS figure conventions: navy header band,
square panels with coloured title bars, flat glyphs, all-caps section labels, key-numbers panel).
Numbers are read from output/tables where available.  Output: output/figures/f0_analysis_schematic.png"""
import os, textwrap, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"]})
NAVY, NAVY2, RULE, INK, GREY, RED = "#16365c", "#2b5d8f", "#9fb3cb", "#1e2a38", "#5d6b7c", "#c0392b"
BLUE, TEAL, ORANGE, PURPLE, GOLD = "#1d5a9e", "#138a7a", "#d0702a", "#7a4b9c", "#d4a017"
T = "output/tables/"

# ---------------- numbers from results (fallback text if a table is not there yet)
d = pd.read_csv("data/meps2022_adults.csv")
n = len(d); mean_tot = np.average(d.total_exp, weights=d.weight)
s_ = d.sort_values("total_exp", ascending=False); cw = s_.weight.cumsum() / s_.weight.sum()
ce = (s_.weight * s_.total_exp).cumsum() / (s_.weight * s_.total_exp).sum(); top5 = float(ce[cw <= 0.05].iloc[-1])
FIND = [(f"\\${mean_tot:,.0f}", "mean annual spending per adult (2022)"),
        (f"Top 5% → {top5:.0%}", "of all spending (highly skewed)")]
try:
    prof = pd.read_csv(T + "u_cluster_profiles.csv", index_col=0)
    FIND.append(("7 profiles", r"from \$2.9k to \$16.5k per person (K-means)"))
except Exception: pass
try:
    m = pd.read_csv(T + "s_regression_metrics.csv"); col = [c for c in m.columns if "r2" in c.lower()][0]
    b = m.loc[m[col].idxmax()]; FIND.append((f"Test R² {b[col]:.2f}", "best model (gradient boosting), log total spending" if b.iloc[0]=="GradBoost" else f"best model ({b.iloc[0]}), log total spending"))
except Exception: pass
try:
    dm = pd.read_csv(T + "i_dml_results.csv"); FIND.append(("DML θ ± 95% CI", "conditional associations, 8 demographic contrasts"))
except Exception: pass

W, H = 17.0, 10.0
fig = plt.figure(figsize=(W, H), dpi=240); fig.patch.set_facecolor("white")
A = fig.add_axes([0, 0, 1, 1]); A.set_xlim(0, W); A.set_ylim(0, H); A.axis("off")

def panel(x, y, w, h, title, col=NAVY, body="white", hh=0.48, fs=12):
    A.add_patch(Rectangle((x, y), w, h, fc=body, ec=col, lw=1.4, zorder=1))
    A.add_patch(Rectangle((x, y + h - hh), w, hh, fc=col, ec=col, lw=1.4, zorder=2))
    A.text(x + 0.17, y + h - hh / 2, title, fontsize=fs, fontweight="bold", color="white", va="center", zorder=3)
def down_arrow(x, y0, y1):
    A.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="simple,head_width=0.9,head_length=0.7,tail_width=0.35",
                                mutation_scale=18, fc=NAVY2, ec="none", zorder=4))
def t_(ax, x, y, s, c=INK, fs=8.5, **k): ax.text(x, y, s, fontsize=fs, color=c, ha=k.pop("ha", "center"), va=k.pop("va", "center"), zorder=8, **k)

# ---------------- glyphs (axis 0-10 x 0-5)
rng = np.random.default_rng(3)
def g_data(ax, col):
    for i, (lab, w_) in enumerate((("persons", 22431), ("adults 18+", n))):
        y = 3.4 - i * 1.7; ax.add_patch(Rectangle((0.5, y), 9 * w_ / 22431, 0.9, fc=col if i else "#c9d6e6", ec="none"))
        t_(ax, 0.7, y + 0.45, f"{w_:,} {lab}", "white" if i else INK, 8.5, ha="left", fontweight="bold")
    t_(ax, 5, 0.6, "weights · strata · PSU", GREY, 8)
def g_desc(ax, col):
    x = np.linspace(0.6, 9.4, 26); h = 4.2 * np.exp(-x / 1.4) + 0.12
    for xi, hi in zip(x, h): ax.add_patch(Rectangle((xi, 0.6), 0.3, hi, fc=col, ec="none"))
    ax.plot([0.5, 9.6], [0.6, 0.6], color=INK, lw=0.8); t_(ax, 6.2, 3.2, "zeros + long right tail", GREY, 8)
def g_unsup(ax, col):
    for k_, (cx, cy, c) in enumerate(((2.3, 3.2, BLUE), (6.0, 3.6, ORANGE), (4.2, 1.4, TEAL), (8.0, 1.6, PURPLE))):
        p = rng.normal([cx, cy], 0.55, (40, 2)); ax.scatter(p[:, 0], p[:, 1], s=7, color=c, alpha=.8, lw=0)
        ax.add_patch(Circle((cx, cy), 1.25, fc="none", ec=c, lw=1, ls=(0, (2, 1.5))))
    ax.annotate("", (9.4, 4.7), (0.5, 0.2), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=.8)); t_(ax, 8.6, 4.65, "PC1", GREY, 7.5, ha="right")
def g_sup(ax, col):
    P = {0: (5, 4.4), 1: (2.8, 2.7), 2: (7.2, 2.7), 3: (1.6, 0.9), 4: (4.0, 0.9), 5: (6.0, 0.9), 6: (8.4, 0.9)}
    for i, j in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]: ax.plot(*zip(P[i], P[j]), color=GREY, lw=1)
    for k_, (u, v) in P.items(): ax.add_patch(Rectangle((u - .35, v - .25), .7, .5, fc=col if k_ < 3 else "#e9b28c", ec="none", zorder=3))
    t_(ax, 5, 0.1, "y = log(1 + $)", GREY, 7.5)
def g_inf(ax, col):
    ax.plot([5, 5], [0.3, 4.7], color=GREY, lw=.8, ls=(0, (3, 2)))
    for i, (e, w_) in enumerate(((-2.2, .7), (1.6, .9), (-0.9, .5), (2.8, .6), (0.6, .8))):
        y = 4.2 - i * 0.9; ax.plot([5 + e - w_, 5 + e + w_], [y, y], color=col, lw=1.4); ax.plot(5 + e, y, "s", color=col, ms=5)
    t_(ax, 5, -0.1, "θ with 95% CI", GREY, 7.5)
def g_shap(ax, col):
    for i, v in enumerate((4.2, 3.75, 3.05, 2.75, 2.7)):
        y = 4.1 - i * 0.85; ax.add_patch(Rectangle((2.6, y), v * 1.6, 0.55, fc=col, ec="none", alpha=1 - i * .13))
    for i, l in enumerate(("chronic dx", "usual source", "family size", "race", "education")): t_(ax, 2.4, 4.37 - i * .85, l, INK, 7.5, ha="right")
    t_(ax, 6.5, 0, "mean |SHAP|", GREY, 7.5)

GROUPS = [
    ("Data & Sample", BLUE, g_data, "MEPS HC-243 (2022) person file; adults 18+ with survey weights; MEPS missing codes recoded."),
    ("Descriptives", BLUE, g_desc, "Survey-weighted means by age, sex, race, income, insurance and region with design-based 95% CIs."),
    ("Unsupervised Learning", TEAL, g_unsup, "PCA + K-means (K = 7), checked with Gaussian mixture and Ward clustering; profile spending."),
    ("Supervised Prediction", ORANGE, g_sup, "LASSO, Ridge, Elastic Net, KNN, random forest, boosting, XGBoost, MLP; two-part model; 5-fold CV."),
    ("Interpretation", ORANGE, g_shap, "SHAP (TreeExplainer), permutation importance, LASSO path and LMG R² decomposition."),
    ("Inference: Double ML", PURPLE, g_inf, "Survey-weighted OLS vs. DML partially linear model with cross-fitting; AIPW for uninsured."),
]

# ---------------- header band
A.add_patch(Rectangle((0, H - 0.95), W, 0.95, fc=NAVY, ec="none"))
A.text(0.35, H - 0.475, "Healthcare Expenditures and Demographics: A Machine Learning Research Portfolio", fontsize=19,
       fontweight="bold", color="white", va="center")
logo = plt.imread("assets/emory_logo_white.png"); lh = 0.42; lw = lh * logo.shape[1] / logo.shape[0]
la = fig.add_axes([(W - 0.35 - lw) / W, (H - 0.475 - lh / 2) / H, lw / W, lh / H]); la.imshow(logo); la.axis("off")

LX0, LX1, TOP, BOT = 0.35, 12.45, H - 1.55, 0.75
RX, RW = 12.75, W - 12.75 - 0.35
for x0_, x1_, lab in ((LX0, LX1, "HOW THE ANALYSIS WAS DONE"), (RX, RX + RW, "FROM DATA TO FINDINGS")):
    A.text(x0_, TOP + 0.3, lab, fontsize=11.5, fontweight="bold", color=NAVY, va="center")
    A.plot([x0_, x1_], [TOP + 0.1] * 2, color=NAVY, lw=1.6)
gx, gy = 0.2, 0.25; pw = (LX1 - LX0 - 2 * gx) / 3; ph = (TOP - BOT - gy) / 2
for i, (name, col, draw, txt) in enumerate(GROUPS):
    r, c = divmod(i, 3); x = LX0 + c * (pw + gx); y = TOP - (r + 1) * ph - r * gy
    panel(x, y, pw, ph, f"{i + 1}  {name}", col)
    dax = fig.add_axes([(x + 0.35) / W, (y + 1.55) / H, (pw - 0.7) / W, (ph - 2.15) / H]); dax.set_xlim(0, 10); dax.set_ylim(-0.3, 5)
    dax.axis("off"); draw(dax, col)
    A.text(x + 0.2, y + 1.3, textwrap.fill(txt, 40, break_on_hyphens=False), fontsize=12, color=INK, va="top", linespacing=1.35)

# ---------------- right: flow
def box(y0, y1, title, sub, col=NAVY2):
    A.add_patch(Rectangle((RX, y0), RW, y1 - y0, fc="white", ec=col, lw=1.4, zorder=2))
    A.add_patch(Rectangle((RX, y1 - 0.38), RW, 0.38, fc=col, ec=col, zorder=2))
    A.text(RX + 0.17, y1 - 0.19, title, fontsize=10.5, fontweight="bold", color="white", va="center", zorder=3)
    A.text(RX + 0.17, (y0 + y1 - 0.38) / 2, sub, fontsize=9.4, color=INK, va="center", linespacing=1.3, zorder=3)
b1 = (TOP - 1.05, TOP); box(*b1, "Outcomes", "Total & out-of-pocket spending, any use,\nlog(1 + $), top-10% high-cost user")
down_arrow(RX + RW / 2, b1[0] - 0.04, b1[0] - 0.26)
b2 = (b1[0] - 1.35, b1[0] - 0.3); box(*b2, "Predictors", "Age, sex, race/ethnicity, education, income\n(% FPL), family size, region; insurance & health")
down_arrow(RX + RW / 2, b2[0] - 0.04, b2[0] - 0.26)
my1 = b2[0] - 0.3; my0 = my1 - 0.5; bw = (RW - 0.2) / 3
for k_, (lab, col) in enumerate((("Cluster", TEAL), ("Predict", ORANGE), ("Infer", PURPLE))):
    A.add_patch(Rectangle((RX + k_ * (bw + 0.1), my0), bw, my1 - my0, fc=col, ec="none"))
    A.text(RX + k_ * (bw + 0.1) + bw / 2, (my0 + my1) / 2, lab, fontsize=11, fontweight="bold", color="white", ha="center", va="center")
A.add_patch(FancyArrowPatch((LX1 + 0.02, (my0 + my1) / 2), (RX - 0.02, (my0 + my1) / 2), arrowstyle="simple,head_width=0.9,head_length=0.7,tail_width=0.35",
                            mutation_scale=18, fc=NAVY2, ec="none", zorder=4))
down_arrow(RX + RW / 2, my0 - 0.05, my0 - 0.27)
sy0, sy1 = BOT, my0 - 0.32
panel(RX, sy0, RW, sy1 - sy0, "KEY NUMBERS", NAVY, body="#f3f6fa")
FN = 0.3; step = (sy1 - 0.48 - sy0 - FN) / max(len(FIND), 4); yy = sy1 - 0.48 - step / 2 - 0.03
for k_, (st, h_) in enumerate(FIND):
    A.text(RX + 0.22, yy + 0.12, st, fontsize=16, fontweight="bold", color=RED, va="center")
    A.text(RX + 0.22, yy - 0.17, h_, fontsize=9.8, color=NAVY, va="center")
    if k_ < len(FIND) - 1: A.plot([RX + 0.18, RX + RW - 0.18], [yy - step / 2] * 2, color=RULE, lw=0.8)
    yy -= step
A.text(RX + 0.22, sy0 + 0.15, "Cross-sectional: associations, not causal effects", fontsize=9.5, color=GREY, va="center", style="italic")
import sys; sys.path.insert(0, "code"); import style
import textwrap as _tw
A.text(0.35, 0.36, _tw.fill("Abbreviations: " + style._abbrevs(fig), 250), fontsize=8.5, color=GREY, va="center", ha="left", linespacing=1.4)
fig.savefig("output/figures/f0_analysis_schematic.png", facecolor="white")
print(FIND)
