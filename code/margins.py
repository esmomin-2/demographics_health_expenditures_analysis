"""Health-adjusted group means via regression standardization (predictive margins).
Fit survey-weighted WLS  y ~ C(group) [* C(by)] + controls, SEs clustered by strata x PSU; for each level set everyone's group
to that level, predict, take the weighted mean. Linear in beta, so SE = sqrt(L V L')."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
HEALTH = "C(health_poor) + C(mental_poor) + diabetes + hypertension + heart_disease + asthma + arthritis + cancer"
NEED = ["health_poor", "mental_poor", "diabetes", "hypertension", "heart_disease", "asthma", "arthritis", "cancer"]

def adjusted_means(d, y, group, controls, by=None):
    d = d.dropna(subset=NEED + [y, group] + ([by] if by else [])).copy(); d["_grp"] = d.strata * 100 + d.psu
    rhs = f"C({group})" + (f" * C({by})" if by else "") + " + " + controls
    m = smf.wls(f"{y} ~ {rhs}", d, weights=d.weight).fit(cov_type="cluster", cov_kwds={"groups": d._grp})
    V = m.cov_params().values; w = d.weight.values / d.weight.sum(); out = []
    from patsy import build_design_matrices
    di = m.model.data.design_info
    for b in (sorted(d[by].unique()) if by else [None]):
        sub = d if b is None else d[d[by] == b]; ws = sub.weight.values / sub.weight.sum()
        for g in sorted(d[group].unique()):
            dd = sub.copy(); dd[group] = g
            X = np.asarray(build_design_matrices([di], dd)[0]); L = ws @ X
            out.append(dict(by=b, level=g, adj_mean=float(L @ m.params.values), se=float(np.sqrt(L @ V @ L))))
    return pd.DataFrame(out)
