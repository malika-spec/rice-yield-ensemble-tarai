"""
STAGE 5 direction-table fix (run once, after stage 5).
Run:    python stage5_direction_fix.py
Why:    stage 5 computed Spearman correlations on hold-out rows that contain
        missing humidity values, so humidity correlations came out blank.
        This script drops missing values pairwise for the correlation only.
What it does:
  1. Recomputes the SHAP values with the same seeded explainers as stage 5.
  2. Checks the mean |SHAP| values against results/stage5_shap.csv.
  3. Writes results/stage5_shap_direction.csv only if the check passes.
"""
import warnings; warnings.filterwarnings("ignore")
import json, sys
import numpy as np, pandas as pd, shap
from scipy.stats import spearmanr
from common import *

D = load_data()
S1 = load_json(os.path.join(OUT, "stage1_params.json"))
assert S1["data_sha"] == D.data_sha, "dataset mismatch with stage 1"
P = S1["params"]
S2 = load_json(os.path.join(OUT, "stage2_selection.json"))
PV = S2["primary_voting"]
baselines, ensembles = candidates(P)
ALL = {**baselines, **ensembles}

Xte = pd.DataFrame(D.X_te.values, columns=D.features)
Xtr = pd.DataFrame(D.X_tr.values, columns=D.features)

# same steps, same order, as stage 5
cat = make_pipe(build_cat(P["CatBoost"])).fit(D.X_tr, D.y_tr)
sv_cat = shap.TreeExplainer(cat.named_steps["clf"]).shap_values(cat[:-1].transform(Xte))
sv_cat = sv_cat[1] if isinstance(sv_cat, list) else sv_cat
prim = ALL[PV]().fit(D.X_tr, D.y_tr)
bg = shap.sample(Xtr, 50, random_state=RANDOM_STATE)
np.random.seed(RANDOM_STATE)
kex = shap.KernelExplainer(lambda Z: prim.predict_proba(pd.DataFrame(Z, columns=D.features))[:, 1], bg)
sv_ens = kex.shap_values(Xte, nsamples=SHAP_NSAMPLES, silent=True)

# 2. consistency with the stage 5 importance table
s5 = pd.read_csv(os.path.join(OUT, "stage5_shap.csv"), index_col=0)
ens_col = [c for c in s5.columns if "KernelSHAP" in c][0]
cat_col = [c for c in s5.columns if "TreeSHAP" in c][0]
mismatch = []
for f in D.features:
    j = D.features.index(f)
    for col, sv in [(cat_col, sv_cat), (ens_col, sv_ens)]:
        if abs(round(float(np.abs(sv[:, j]).mean()), 4) - float(s5.loc[f, col])) > 0.00015:
            mismatch.append((f, col))
if mismatch:
    print("STOP: SHAP means differ from stage5_shap.csv for:", mismatch[:5])
    print("Do not use the direction table. Send this message to the author.")
    sys.exit(1)
print("SHAP means match stage5_shap.csv for all 18 variables and both models.")

# 3. direction table. Missing humidity is handled two ways:
#    (a) missing = 0            (the rule chosen for the manuscript)
#    (b) missing = training mean (the value the model actually used, after imputation)
#    Both are reported so the editors can see whether the conclusion depends on the rule.
train_mean_h = float(np.nanmean(D.X_tr["humidity"].values))
print(f"missing humidity in test rows: {int(np.isnan(D.X_te['humidity'].values).sum())} of {len(D.X_te)}")
rows = []
for f in ["humidity", "rain_1h", "Area (Ha)", "pressure", "wind_speed"]:
    j = D.features.index(f)
    x_raw = D.X_te.values[:, j].astype(float)
    variants = [("missing=0", np.where(np.isnan(x_raw), 0.0, x_raw))]
    if f == "humidity":
        variants.append(("missing=training mean", np.where(np.isnan(x_raw), train_mean_h, x_raw)))
    for vname, x in variants:
        for name, sv in [("CatBoost (TreeSHAP)", sv_cat), (f"{PV} (KernelSHAP)", sv_ens)]:
            rho, p = spearmanr(x, sv[:, j])
            rows.append(dict(feature=f, handling_of_missing=vname, model=name,
                             spearman_rho=round(float(rho), 3), p_value=round(float(p), 4),
                             n_rows=len(x), mean_abs_shap=round(float(np.abs(sv[:, j]).mean()), 4)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage5_shap_direction.csv"), index=False)
print("written: results/stage5_shap_direction.csv")
