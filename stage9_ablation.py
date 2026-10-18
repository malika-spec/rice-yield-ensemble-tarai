"""
STAGE 9 — ablation of tuning, ensemble membership and soil variables.
Run:    python stage9_ablation.py
Needs:  stage1_params.json, dataset, common.py (same folder)
Writes ONLY new files: results/stage9_ablation.csv, results/stage9_meta.json
Repeated CV is run on the 107 training rows only. The hold-out (47 rows) is scored
once per variant and reported descriptively; it is not used to choose anything.
"""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np, pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import VotingClassifier
from common import *

t0 = time.time()
D = load_data()
S1 = load_json(os.path.join(OUT, "stage1_params.json"))
assert S1["data_sha"] == D.data_sha, "dataset mismatch with stage 1"
P = S1["params"]
DEF = {"CatBoost": {}, "XGBoost": {}, "LightGBM": {}}   # library defaults (no tuning)

SOIL = [f for f in D.features if any(k in f for k in
        ["Soil", "EC", "Available", "Mineralizable"])]
assert len(SOIL) == 7, f"expected 7 soil variables, found {SOIL}"
NOSOIL = [f for f in D.features if f not in SOIL]

def vote(estimators, weights, PP):
    return make_pipe(VotingClassifier(estimators=estimators, voting="soft",
                                      weights=weights, n_jobs=1))

def members(PP, keep):
    allm = {"CatBoost": ("cat", build_cat(PP["CatBoost"])),
            "XGBoost":  ("xgb", build_xgb(PP["XGBoost"])),
            "LightGBM": ("lgb", build_lgb(PP["LightGBM"]))}
    return [allm[k] for k in keep]

VARIANTS = {
    # name: (estimator factory, feature list)
    "Tuned CatBoost (reference)":        (lambda: make_pipe(build_cat(P["CatBoost"])), "all"),
    "Default CatBoost (no tuning)":      (lambda: make_pipe(build_cat(DEF["CatBoost"])), "all"),
    "Tuned voting [5,1,1] (principal)":  (lambda: vote(members(P, ["CatBoost","XGBoost","LightGBM"]), [5,1,1], P), "all"),
    "Default voting [5,1,1] (no tuning)":(lambda: vote(members(DEF, ["CatBoost","XGBoost","LightGBM"]), [5,1,1], DEF), "all"),
    "Tuned voting, CatBoost removed [1,1]":(lambda: vote(members(P, ["XGBoost","LightGBM"]), [1,1], P), "all"),
    "Tuned voting, XGBoost removed [5,1]": (lambda: vote(members(P, ["CatBoost","LightGBM"]), [5,1], P), "all"),
    "Tuned voting, LightGBM removed [5,1]":(lambda: vote(members(P, ["CatBoost","XGBoost"]), [5,1], P), "all"),
    "Tuned voting [5,1,1], no soil variables": (lambda: vote(members(P, ["CatBoost","XGBoost","LightGBM"]), [5,1,1], P), "nosoil"),
    "Tuned CatBoost, no soil variables": (lambda: make_pipe(build_cat(P["CatBoost"])), "nosoil"),
}

rows = []
for name, (factory, feats) in VARIANTS.items():
    cols = D.features if feats == "all" else NOSOIL
    Xtr, Xte = D.X_tr[cols], D.X_te[cols]
    s = cross_validate(factory(), Xtr, D.y_tr, cv=repeated_cv(), scoring="accuracy", n_jobs=1)
    fold = s["test_score"]
    fit = factory().fit(Xtr, D.y_tr)
    ho = float((fit.predict(Xte) == D.y_te.values).mean())
    rows.append(dict(variant=name, n_features=len(cols),
                     repeated_cv_acc=round(float(fold.mean()), 4),
                     repeated_cv_sd_population=round(float(np.std(fold)), 4),
                     holdout_acc_descriptive=round(ho, 4)))
    print(f"{name:<42} CV {fold.mean()*100:5.1f} (SD {np.std(fold)*100:4.1f})  hold-out {ho*100:5.1f}", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage9_ablation.csv"), index=False)
save_json(dict(stage="stage9", data_sha=D.data_sha, params_sha=S1["params_sha"],
               soil_variables=SOIL, versions=versions(),
               seconds=round(time.time() - t0, 1)), os.path.join(OUT, "stage9_meta.json"))
print("stage 9 done", flush=True)
