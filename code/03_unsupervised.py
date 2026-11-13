"""Unsupervised profiles of US adults (MEPS 2022): PCA, K-means, GMM, Ward; weighted spending profiles."""
import sys; sys.path.insert(0, "code"); import style; style.apply()
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

RS = 42
UNW = "Source: MEPS 2022 Full-Year Consolidated File (HC-243); adults 18+, unweighted."
FIG, TAB = "output/figures/", "output/tables/"
d = pd.read_csv("data/meps2022_adults.csv")

num = ["age", "female", "educ_yrs", "fam_size", "health_poor", "mental_poor", "married", "employed",
       "has_usual_source", "diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer", "n_chronic"]
cat = ["race", "pov_cat", "insurance", "region"]
Xn = d[num].fillna(d[num].median())
X = pd.concat([Xn, pd.get_dummies(d[cat], prefix=cat, prefix_sep=": ", dtype=float)], axis=1)
Z = StandardScaler().fit_transform(X)
feat = list(X.columns)

# ---- PCA
pca = PCA(random_state=RS).fit(Z)
P = pca.transform(Z)
ev = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(8, 4.5))
k = np.arange(1, len(ev) + 1)
ax.bar(k, ev * 100, color=style.NAVY, label="Individual")
ax.plot(k, np.cumsum(ev) * 100, "o-", color=style.RUST, ms=3, label="Cumulative")
ax.set_xlabel("Principal component (index)"); ax.set_ylabel("Variance explained (%)")
ax.set_title("PCA explained variance"); ax.legend(); fig.tight_layout()
style.save(fig, "u_pca_scree", UNW)
pd.DataFrame({"PC": k, "explained_var": ev, "cumulative": np.cumsum(ev)}).to_csv(TAB + "u_pca_variance.csv", index=False)
load = pd.DataFrame(pca.components_[:3].T, index=feat, columns=["PC1", "PC2", "PC3"])
load.round(3).to_csv(TAB + "u_pca_loadings.csv")
for pc in ["PC1", "PC2", "PC3"]:
    print(pc, load[pc].sort_values(key=abs, ascending=False).head(6).round(2).to_dict())

rng = np.random.default_rng(RS)
sub = rng.choice(len(d), 5000, replace=False)
fig, ax = plt.subplots(figsize=(8, 6.5))
sc = ax.scatter(P[sub, 0], P[sub, 1], c=d.log_total_exp.values[sub], s=4, alpha=.35, linewidths=0, cmap=style.CMAP)
plt.colorbar(sc, label="Total expenditure, log(1 + $)")
top = load[["PC1", "PC2"]].pow(2).sum(1).sort_values(ascending=False).index[:10]
s = 6
for f in top:
    ax.arrow(0, 0, load.loc[f, "PC1"] * s, load.loc[f, "PC2"] * s, color=style.RUST, head_width=.08)
    ax.text(load.loc[f, "PC1"] * s * 1.12, load.loc[f, "PC2"] * s * 1.12, f, fontsize=8, color=style.RUST)
ax.set_xlabel(f"PC1 score (SD units; {ev[0]*100:.1f}% of variance)"); ax.set_ylabel(f"PC2 score (SD units; {ev[1]*100:.1f}% of variance)")
ax.set_title("PCA biplot colored by total spending"); fig.tight_layout()
style.save(fig, "u_pca_biplot", UNW + " Random subsample, n = 5,000; loading arrows scaled x6.")

# ---- K selection
Ks = range(2, 11); inert, sil = [], []
for K in Ks:
    km = KMeans(K, n_init=10, random_state=RS).fit(Z)
    inert.append(km.inertia_); sil.append(silhouette_score(Z[sub], km.labels_[sub], random_state=RS))
ksel = pd.DataFrame({"K": list(Ks), "inertia": inert, "silhouette": sil}); ksel.to_csv(TAB + "u_kmeans_selection.csv", index=False)
print(ksel.round(4))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ksel.K, ksel.inertia / 1e3, "o-", color=style.NAVY); ax[0].set(xlabel="Number of clusters (K)", ylabel="Inertia (within-cluster SS, thousands)", title="Elbow")
ax[1].plot(ksel.K, ksel.silhouette, "o-", color=style.TEAL); ax[1].set(xlabel="Number of clusters (K)", ylabel="Silhouette score (unitless, -1 to 1)", title="Silhouette (5,000 subsample)")
fig.tight_layout(); style.save(fig, "u_kmeans_selection", UNW)

K = int(ksel[ksel.K.between(4, 7)].sort_values("silhouette", ascending=False).K.iloc[0])
print("Chosen K:", K)
km = KMeans(K, n_init=20, random_state=RS).fit(Z); lab = km.labels_
gmm = GaussianMixture(K, covariance_type="diag", random_state=RS, n_init=3).fit(Z)
ward = AgglomerativeClustering(K, linkage="ward").fit(Z[sub])
agree = pd.DataFrame({"comparison": ["K-means vs GMM (full sample)", "K-means vs Ward (5k subsample)"],
                      "ARI": [adjusted_rand_score(lab, gmm.predict(Z)), adjusted_rand_score(lab[sub], ward.labels_)]})
agree.to_csv(TAB + "u_cluster_agreement.csv", index=False); print(agree)

# ---- Weighted profiles
d["cluster"] = lab
w = d.weight
prof_vars = feat + ["total_exp", "oop_exp", "any_exp", "oop_share_income", "office_visits", "er_visits"]
D = pd.concat([X, d[["total_exp", "oop_exp", "any_exp", "oop_share_income", "office_visits", "er_visits", "cluster", "weight"]]], axis=1)
def wmean(g):
    out = {}
    for v in prof_vars:
        m = g[v].notna(); out[v] = np.average(g.loc[m, v], weights=g.loc[m, "weight"])
    out["pop_share"] = g.weight.sum() / w.sum(); out["n"] = len(g)
    return pd.Series(out)
prof = D.groupby("cluster").apply(wmean).T
print(prof.loc[["pop_share", "n", "age", "health_poor", "n_chronic", "insurance: Public only", "insurance: Uninsured",
                "pov_cat: High income", "pov_cat: Poor (<100% FPL)", "employed", "married", "race: Hispanic", "female",
                "has_usual_source", "educ_yrs", "total_exp", "oop_exp", "any_exp", "oop_share_income"]].round(3).to_string())

import json, sys
names = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {i: f"Cluster {i}" for i in range(K)}
names = {int(k): v for k, v in names.items()}
prof.columns = [f"{c}: {names[c]}" for c in prof.columns]
prof.round(4).to_csv(TAB + "u_cluster_profiles.csv")
d["cname"] = d.cluster.map(names)

# ---- Weighted regression of log spending on cluster
d["grp"] = d.strata * 100 + d.psu
ref = d.groupby("cname").weight.sum().idxmax()
mod = smf.wls(f"log_total_exp ~ C(cname, Treatment(reference='{ref}'))", data=d, weights=d.weight).fit(
    cov_type="cluster", cov_kwds={"groups": d.grp})
ct = pd.DataFrame({"coef": mod.params, "se": mod.bse, "p": mod.pvalues}).round(4)
ct.index = [i.replace(f"C(cname, Treatment(reference='{ref}'))[T.", "").rstrip("]") for i in ct.index]
ct.to_csv(TAB + "u_cluster_logexp_wls.csv")
print("ref:", ref); print(ct)
R = np.zeros((K - 1, K)); R[:, 1:] = np.eye(K - 1)
print("Joint Wald:", mod.wald_test(R, scalar=True))

# ---- Figures: spending by cluster, embeddings colored by cluster
CL = ["#1D4E89", "#E4572E", "#17A398", "#F3A712", "#8E44AD", "#6B8E23", "#D81E5B", "#00A6ED", "#7F5539", "#555555"]
s2 = pd.DataFrame({"Mean total": prof.loc["total_exp"], "Mean OOP": prof.loc["oop_exp"]})
s2.index = [names[i] for i in range(K)]
fig, ax = plt.subplots(figsize=(9, 4.5))
y = np.arange(K)
ax.barh(y - .2, s2["Mean total"], .4, color=[CL[i] for i in range(K)], label="Mean total")
ax.barh(y + .2, s2["Mean OOP"], .4, color=[CL[i] for i in range(K)], alpha=.45, label="Mean out-of-pocket (lighter)")
ax.set_yticks(y, s2.index); ax.invert_yaxis(); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Mean expenditure per person, 2022 (US $)"); ax.set_title("Healthcare spending by profile"); ax.legend()
fig.tight_layout(); style.save(fig, "u_cluster_spending")
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import umap
Zs, ls = Z[sub], lab[sub]
EMB = {"PCA": P[sub][:, :2]}
print("t-SNE ..."); EMB["t-SNE"] = TSNE(2, perplexity=40, init="pca", learning_rate="auto", random_state=RS).fit_transform(Zs)
print("UMAP ..."); EMB["UMAP"] = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=RS).fit_transform(Zs)
AX = {"PCA": ("PC1 (SD units)", "PC2 (SD units)"), "t-SNE": ("t-SNE 1 (arbitrary units)", "t-SNE 2 (arbitrary units)"),
      "UMAP": ("UMAP 1 (arbitrary units)", "UMAP 2 (arbitrary units)")}

def draw(ax, E, title):
    lo, hi = np.percentile(E, [0.5, 99.5], axis=0); pad = (hi - lo) * 0.06; lo, hi = lo - pad, hi + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 220), np.linspace(lo[1], hi[1], 220))
    for c in range(K):
        m = ls == c; ax.scatter(E[m, 0], E[m, 1], s=5, alpha=.55, linewidths=0, color=CL[c], zorder=2)
    for c in range(K):                                   # boundary: 60% highest-density region of each profile
        m = ls == c; k_ = gaussian_kde(E[m].T); dens = k_(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        lev = np.quantile(k_(E[m].T), 0.40)
        ax.contourf(gx, gy, dens, levels=[lev, dens.max() + 1], colors=[CL[c]], alpha=.10, zorder=1)
        ax.contour(gx, gy, dens, levels=[lev], colors=[CL[c]], linewidths=1.6, zorder=3)
        cx, cy = np.median(E[m], axis=0)
        ax.text(cx, cy, str(c + 1), fontsize=9, fontweight="bold", color="white", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.25", fc=CL[c], ec="white", lw=1))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.grid(False)
    ax.set_xlabel(AX[title][0]); ax.set_ylabel(AX[title][1]); ax.set_title(title)
    if title != "PCA": ax.set_xticks([]); ax.set_yticks([])

from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", ms=8, color=CL[c], label=f"{c + 1}  {names[c]}") for c in range(K)]
fig, axs = plt.subplots(1, 3, figsize=(17, 6.4))
for a, t in zip(axs, EMB): draw(a, EMB[t], t)
axs[0].set_title("PCA (linear)"); axs[1].set_title("t-SNE (local structure)"); axs[2].set_title("UMAP (local + global)")
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.legend(handles=handles, loc="upper left", ncol=4, fontsize=10, bbox_to_anchor=(0.035, 0.995), frameon=False,
           columnspacing=2.2, handletextpad=0.4, borderaxespad=0)
fig.text(0.035, 0.0, "", fontsize=1)
style.save(fig, "u_embeddings_compare", UNW + " Same 5,000 adults and K-means labels in all panels; outlines = densest 60% of each profile.")
from sklearn.metrics import silhouette_score as ss
print({t: round(ss(E, ls), 3) for t, E in EMB.items()})
