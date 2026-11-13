"""01 — Build analysis sample from MEPS HC-243 (2022 Full-Year Consolidated File).
Unit: adult (18+) person with positive person weight. Negative MEPS codes (-1,-7,-8,-15) -> missing.
"""
import numpy as np, pandas as pd
raw = pd.read_stata("data/h243.dta", convert_categoricals=False)
keep = ["DUPERSID","DUID","FAMIDYR","AGE22X","SEX","RACETHX","EDUCYR","HIDEG","POVCAT22","FAMINC22",
        "FAMSZEYR","INSCOV22","RTHLTH53","MNHLTH53","REGION22","MARRY22X","EMPST53","HAVEUS42",
        "DIABDX_M18","HIBPDX","CHDDX","ASTHDX","ARTHDX","CANCERDX",
        "TOTEXP22","TOTSLF22","OBTOTV22","ERTOT22","IPDIS22","RXTOT22","PERWT22F","VARSTR","VARPSU"]
d = raw[keep].copy()
d = d[(d.AGE22X >= 18) & (d.PERWT22F > 0)]
for c in ["EDUCYR","HIDEG","RTHLTH53","MNHLTH53","REGION22","MARRY22X","EMPST53","HAVEUS42",
          "DIABDX_M18","HIBPDX","CHDDX","ASTHDX","ARTHDX","CANCERDX"]:
    d.loc[d[c] < 0, c] = np.nan
out = pd.DataFrame({
    "dupersid": d.DUPERSID, "duid": d.DUID,
    "age": d.AGE22X, "female": (d.SEX == 2).astype(int),
    "race": d.RACETHX.map({1:"Hispanic",2:"NH White",3:"NH Black",4:"NH Asian",5:"NH Other/Multi"}),
    "educ_yrs": d.EDUCYR,
    "pov_cat": d.POVCAT22.map({1:"Poor (<100% FPL)",2:"Near poor",3:"Low income",4:"Middle income",5:"High income"}),
    "fam_income": d.FAMINC22.clip(lower=0),
    "fam_size": d.FAMSZEYR,
    "insurance": d.INSCOV22.map({1:"Any private",2:"Public only",3:"Uninsured"}),
    "health_poor": d.RTHLTH53,          # 1 excellent ... 5 poor
    "mental_poor": d.MNHLTH53,
    "region": d.REGION22.map({1:"Northeast",2:"Midwest",3:"South",4:"West"}),
    "married": (d.MARRY22X == 1).astype(float).where(d.MARRY22X.notna()),
    "employed": d.EMPST53.isin([1,2,3]).astype(float).where(d.EMPST53.notna()),
    "has_usual_source": (d.HAVEUS42 == 1).astype(float).where(d.HAVEUS42.notna()),
})
for nm, c in {"diabetes":"DIABDX_M18","hypertension":"HIBPDX","heart_disease":"CHDDX","asthma":"ASTHDX",
              "arthritis":"ARTHDX","cancer":"CANCERDX"}.items():
    out[nm] = (d[c] == 1).astype(float).where(d[c].notna())
out["n_chronic"] = out[["diabetes","hypertension","heart_disease","asthma","arthritis","cancer"]].sum(axis=1, min_count=1)
out["total_exp"] = d.TOTEXP22; out["oop_exp"] = d.TOTSLF22
out["any_exp"] = (d.TOTEXP22 > 0).astype(int); out["log_total_exp"] = np.log1p(d.TOTEXP22)
out["log_oop_exp"] = np.log1p(d.TOTSLF22)
out["oop_share_income"] = d.TOTSLF22 / d.FAMINC22.clip(lower=1000)
out["office_visits"]=d.OBTOTV22; out["er_visits"]=d.ERTOT22; out["inpatient_stays"]=d.IPDIS22; out["rx_fills"]=d.RXTOT22
out["weight"] = d.PERWT22F; out["strata"] = d.VARSTR; out["psu"] = d.VARPSU
out = out.dropna(subset=["race","pov_cat","insurance","region"]).reset_index(drop=True)
out.to_csv("data/meps2022_adults.csv", index=False)
print(out.shape); print(out.isna().mean().round(3)[lambda s: s>0])
w = out.weight
print("Weighted mean total exp:", np.average(out.total_exp, weights=w).round(0),
      " OOP:", np.average(out.oop_exp, weights=w).round(0), " %any:", np.average(out.any_exp, weights=w).round(3))
