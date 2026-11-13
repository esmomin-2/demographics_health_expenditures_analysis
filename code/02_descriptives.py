"""02 — Survey-weighted descriptives by demographic group.
Uses MEPS person weights; SEs by Taylor linearization with strata (VARSTR) and PSU (VARPSU), domain estimation."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv("data/meps2022_adults.csv")
d["age_grp"] = pd.cut(d.age, [17,34,49,64,120], labels=["18-34","35-49","50-64","65+"]).astype(str)
d["sex"] = np.where(d.female==1,"Female","Male")

def svy_domain_mean(df, mask, y):
    w = df.weight.values*mask; W = w.sum(); m = np.sum(w*df[y].values)/W
    z = w*(df[y].values-m)/W
    g = pd.DataFrame({"z":z,"h":df.strata,"p":df.psu}).groupby(["h","p"]).z.sum().reset_index()
    var = sum(len(gh)/(len(gh)-1)*((gh.z-gh.z.mean())**2).sum() for _, gh in g.groupby("h") if len(gh)>1)
    return m, np.sqrt(var)

rows=[]
for var in ["age_grp","sex","race","pov_cat","insurance","region"]:
    for lvl in sorted(d[var].unique()):
        mask = (d[var]==lvl).values.astype(float); r={"variable":var,"group":lvl,"n":int(mask.sum())}
        for y in ["total_exp","oop_exp","any_exp"]:
            m,se = svy_domain_mean(d,mask,y); r[y]=m; r[y+"_se"]=se
        rows.append(r)
t = pd.DataFrame(rows); t.round(3).to_csv("output/tables/t1_weighted_means_by_group.csv", index=False)
print(t.round(1).to_string())
# ---- health-adjusted means (regression standardization): control for age, sex and health status
import sys as _s; _s.path.insert(0, "code"); from margins import adjusted_means, HEALTH
d["female_c"] = d.female
arows = []
for var in ["age_grp", "race", "pov_cat", "insurance"]:
    ctrl = " + ".join([c for c in ["C(age_grp)", "female"] if not (var == "age_grp" and "age" in c)] + [HEALTH])
    r_ = {}
    for y in ["total_exp", "oop_exp"]:
        a = adjusted_means(d, y, var, ctrl).set_index("level"); r_[y] = a
    for lv in r_["total_exp"].index:
        arows.append(dict(variable=var, group=lv, total_exp=r_["total_exp"].adj_mean[lv], total_exp_se=r_["total_exp"].se[lv],
                          oop_exp=r_["oop_exp"].adj_mean[lv], oop_exp_se=r_["oop_exp"].se[lv]))
ta = pd.DataFrame(arows); ta.round(2).to_csv("output/tables/t1b_health_adjusted_means_by_group.csv", index=False); print(ta.round(0))
t_unadj = t; t = ta
m,se = svy_domain_mean(d,np.ones(len(d)),"total_exp"); print("Overall total:",round(m),"SE",round(se))

import sys; sys.path.insert(0,"code"); import style; style.apply()
LABELS={"age_grp":"Age group","race":"Race/ethnicity","pov_cat":"Family income (% of FPL)","insurance":"Insurance coverage"}
ORDER={"age_grp":["18-34","35-49","50-64","65+"],"pov_cat":["Poor (<100% FPL)","Near poor","Low income","Middle income","High income"]}
fig, axes = plt.subplots(2,2, figsize=(10.5,6.8))
for ax,var in zip(axes.flat,LABELS):
    s=t[t.variable==var].set_index("group")
    s=s.loc[ORDER[var][::-1] if var in ORDER else s.sort_values("total_exp").index.tolist()]
    y=np.arange(len(s)); h=0.38
    ax.barh(y+h/2, s.total_exp/1e3, h, xerr=1.96*s.total_exp_se/1e3, color=style.NAVY, error_kw=dict(lw=.8,capsize=0,ecolor="#555"), label="Total")
    ax.barh(y-h/2, s.oop_exp/1e3, h, xerr=1.96*s.oop_exp_se/1e3, color=style.RUST, error_kw=dict(lw=.8,capsize=0,ecolor="#555"), label="Out-of-pocket")
    ax.set_yticks(y); ax.set_yticklabels(s.index); ax.set_title(LABELS[var]); ax.grid(axis="y",visible=False); ax.grid(axis="x",visible=True)
    ax.set_xlabel("Health-adjusted mean annual spending ($ thousands)")
    u=t_unadj[t_unadj.variable==var].set_index("group").reindex(s.index)
    ax.plot(u.total_exp/1e3, y+h/2, "D", ms=4.5, mfc="white", mec="#222", mew=1, zorder=5, label="Unadjusted total")
fig.suptitle("Health-adjusted spending by demographic group, 2022", x=0.0, ha="left", fontsize=13, fontweight="bold")
h,l=axes[0,0].get_legend_handles_labels(); fig.tight_layout(rect=(0,0,1,0.955)); fig.legend(h,l,loc="upper right",ncol=3,bbox_to_anchor=(1.0,0.995),frameon=False); style.save(fig,"f1_means_by_group", "Source: MEPS HC-243 (2022), adults 18+. Bars = survey-weighted predictive margins adjusting for age, sex, self-rated physical/mental health and 6 chronic conditions; whiskers = 95% CIs (strata×PSU). Diamonds = unadjusted means.")

fig, ax = plt.subplots(1,3, figsize=(12,3.6))
ax[0].hist(d.total_exp.clip(upper=50000)/1e3, bins=60, color=style.NAVY); ax[0].set_title("Total spending (top-coded at $50k)"); ax[0].set_xlabel("$ thousands"); ax[0].set_ylabel("Respondents")
ax[1].hist(d.log_total_exp, bins=60, color=style.TEAL); ax[1].set_title("log(1 + total spending)"); ax[1].set_xlabel("log $")
s = d.sort_values("total_exp", ascending=False); cw = s.weight.cumsum()/s.weight.sum(); ce=(s.weight*s.total_exp).cumsum()/(s.weight*s.total_exp).sum()
ax[2].plot(cw*100, ce*100, color=style.RUST); ax[2].plot([0,100],[0,100], color=style.GRAY, lw=.8, ls="--")
for p in [.01,.05,.10]:
    v=float(ce[cw<=p].iloc[-1]); ax[2].plot(p*100, v*100, "o", color=style.RUST, ms=4); ax[2].annotate(f"Top {int(p*100)}%: {v:.0%}", (p*100, v*100), xytext=(8,-4), textcoords="offset points", fontsize=8)
ax[2].set_title("Concentration of spending"); ax[2].set_xlabel("% of adults (highest spenders first)"); ax[2].set_ylabel("% of total spending"); ax[2].set_xlim(0,100); ax[2].set_ylim(0,101)
fig.tight_layout(); style.save(fig,"f2_distribution")
print("Top 5% share:", round(float(ce[cw<=0.05].iloc[-1]),3), " Top 1%:", round(float(ce[cw<=0.01].iloc[-1]),3))
