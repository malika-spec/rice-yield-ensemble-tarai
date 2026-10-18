"""
STAGE 8 — confusion matrices for the final candidate models (hold-out, 47 rows).
Run:    python stage8_confusion.py
Needs:  results/stage1_params.json, results/stage4_holdout.csv, dataset, common.py
Writes ONLY new files:
        results/stage8_confusion.csv   TN, FP, FN, TP and derived metrics per model
        results/stage8_meta.json
No stage 1-7 file is modified. Models are refitted with the fixed tuned parameters on the
same 107 training rows; the script stops if any accuracy differs from the stored stage 4 value.
Class coding: 0 = Low, 1 = High. Rows are actual; columns are predicted.
"""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from common import *

t0 = time.time()
D = load_data()
S1 = load_json(os.path.join(OUT, "stage1_params.json"))
assert S1["data_sha"] == D.data_sha, "dataset mismatch with stage 1"
P = S1["params"]
baselines, ensembles = candidates(P)
ALL = {**baselines, **ensembles}
MODELS = ["CatBoost (tuned)", "Weighted Soft Voting [5, 1, 1]",
          "Stacking LR-meta (no-passthrough)", "Stacking RF-meta (no-passthrough)",
          "Hard Voting (3 base)"]
s4 = pd.read_csv(os.path.join(OUT, "stage4_holdout.csv")).set_index("model")

rows = []
for name in MODELS:
    est = ALL[name]().fit(D.X_tr, D.y_tr)
    pred = est.predict(D.X_te)
    y = np.asarray(D.y_te)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y, pred)
    stored = s4.loc[name, "ho_acc"] if name in s4.index else np.nan
    if not np.isnan(stored):
        assert abs(round(acc, 4) - stored) < 0.0001, f"{name}: refit accuracy {acc:.4f} != stored {stored:.4f}"
    sens = tp / (tp + fn); spec = tn / (tn + fp); ppv = tp / (tp + fp); npv = tn / (tn + fn)
    f1 = 2 * ppv * sens / (ppv + sens)
    rows.append(dict(model=name, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                     accuracy=round(acc, 4), sensitivity_High=round(sens, 4),
                     specificity_Low=round(spec, 4), precision_High=round(ppv, 4),
                     NPV_Low=round(npv, 4), F1_High=round(f1, 4)))
    print(f"{name:<36} TN {tn} FP {fp} FN {fn} TP {tp}  acc {acc:.4f}", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage8_confusion.csv"), index=False)
save_json(dict(stage="stage8", data_sha=D.data_sha, params_sha=S1["params_sha"], versions=versions(),
               seconds=round(time.time() - t0, 1)), os.path.join(OUT, "stage8_meta.json"))
print("stage 8 done", flush=True)
