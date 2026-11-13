"""Shared publication figure style: Arial, crisp, minimal. `import style` then style.apply(); save with style.save(fig, name)."""
import matplotlib as mpl, matplotlib.pyplot as plt
PALETTE = ["#1F3A5F", "#C8553D", "#2A9D8F", "#E9A03B", "#6D597A", "#8A8A8A", "#3D7EA6", "#B5838D"]
NAVY, RUST, TEAL, GOLD, PLUM, GRAY = PALETTE[:6]
CMAP = "cividis"
def apply():
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10, "axes.labelcolor": "#222222",
        "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E3E3E3", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "legend.fontsize": 9, "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.6, "patch.linewidth": 0,
    })
EMORY_BLUE = "#16365c"   # CAPS-figure navy
def header(fig, right=None, left="Healthcare Expenditures and Demographics"):
    """CAPS-style navy header band: bold white text left, white Emory wordmark right."""
    import matplotlib.image as mpimg, os
    fig.canvas.draw()
    W, H = fig.get_size_inches(); bh = 0.5 / H
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    x0, x1 = bb.x0 / W - 0.012, bb.x1 / W + 0.012; top = max(1.0, bb.y1 / H) + 0.012
    bar = fig.add_axes([x0, top, x1 - x0, bh]); bar.set_axis_off()
    bar.add_patch(mpl.patches.Rectangle((0, 0), 1, 1, transform=bar.transAxes, fc=EMORY_BLUE, ec="none"))
    bar.text(0.012, 0.5, left, transform=bar.transAxes, ha="left", va="center", color="white", fontsize=10.5, fontweight="bold", clip_on=True)
    logo = mpimg.imread(os.path.join(os.path.dirname(__file__), "..", "assets", "emory_logo_white.png"))
    lh = bh * 0.38; lw = (lh * H) * (logo.shape[1] / logo.shape[0]) / W
    la = fig.add_axes([x1 - 0.012 - lw, top + (bh - lh) / 2, lw, lh]); la.imshow(logo); la.set_axis_off()
ABBR = {"AIPW": "augmented inverse probability weighting", "AUC": "area under the ROC curve", "CI": "confidence interval",
        "CV": "cross-validation", "DML": "double/debiased machine learning", "FPL": "federal poverty line",
        "GMM": "Gaussian mixture model", "KNN": "k-nearest neighbours",
        "LASSO": "least absolute shrinkage and selection operator", "LMG": "Lindeman-Merenda-Gold R² decomposition",
        "MAE": "mean absolute error", "MEPS": "Medical Expenditure Panel Survey", "MLP": "multilayer perceptron",
        "MSE": "mean squared error", "NH": "non-Hispanic", "OLS": "ordinary least squares", "OOP": "out-of-pocket",
        "PC": "principal component", "PCA": "principal component analysis", "pp": "percentage points",
        "PSU": "primary sampling unit", "RMSE": "root mean squared error", "ROC": "receiver operating characteristic",
        "SD": "standard deviation", "SES": "socioeconomic status", "SE": "standard error", "SHAP": "Shapley additive explanations",
        "t-SNE": "t-distributed stochastic neighbour embedding", "UMAP": "uniform manifold approximation and projection",
        "WLS": "weighted least squares", "XGBoost": "extreme gradient boosting"}
def _abbrevs(fig, extra=""):
    import re
    txt = [extra]
    for t in fig.findobj(mpl.text.Text): txt.append(t.get_text())
    blob = " ".join(txt); found = []
    for k in ABBR:
        if re.search(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", blob): found.append(k)
    found = [k for k in found if not (k == "PC" and "PCA" in found and not re.search(r"PC\d", blob))]
    return "; ".join(f"{k}, {ABBR[k]}" for k in sorted(found, key=str.lower))
def save(fig, name, source="Source: MEPS 2022 Full-Year Consolidated File (HC-243); adults 18+, survey-weighted.", banner=True):
    import textwrap, re
    if source:   # keep footers short: data source + at most one method sentence
        source = re.sub(r"MEPS (2022 Full-Year Consolidated File \(HC-243\)|HC-243 \(2022\)|2022 HC-243)", "MEPS 2022", source)
        parts = re.split(r"(?<=\.)\s+", source.strip())
        source = " ".join(parts[:2])
        if len(source) > 190: source = source[:187].rsplit(" ", 1)[0] + "…"
    ab = _abbrevs(fig, source or ""); w = int(fig.get_size_inches()[0] * 17)
    if source: source = textwrap.fill(source, w)
    nl = source.count("\n") + 1 if source else 0
    if source: fig.text(0.0, -0.01, source, fontsize=7.5, color="#666666", ha="left", va="top", transform=fig.transFigure, linespacing=1.3)
    if ab:
        fig.text(0.0, (-0.01 - 0.17 * nl / fig.get_size_inches()[1]) if source else -0.01, textwrap.fill("Abbreviations: " + ab, w), fontsize=7.5, color="#666666",
                 ha="left", va="top", transform=fig.transFigure, linespacing=1.3)
    if banner: header(fig)
    fig.savefig(f"output/figures/{name}.png"); plt.close(fig)
