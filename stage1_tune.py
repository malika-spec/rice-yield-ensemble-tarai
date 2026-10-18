"""
STAGE 1 — Optuna tuning (training data only).
Run:    python stage1_tune.py
Output: results/stage1_params.json
Time:   about 4 minutes on a single core (CatBoost dominates).
"""
import time
from common import *

t0 = time.time()
D = load_data()
print(f"data sha={D.data_sha}  training rows={len(D.X_tr)}  hold-out rows={len(D.X_te)}", flush=True)

params, cv_acc, n_trials = {}, {}, {}
for name in ["CatBoost", "XGBoost", "LightGBM"]:
    p, v, n = tune_on(name, D.X_tr, D.y_tr)
    params[name], cv_acc[name], n_trials[name] = p, v, n
    print(f"{name}: {n} trials, training 5-fold CV accuracy {v:.4f}", flush=True)

out = dict(stage="stage1_tune", data_sha=D.data_sha, versions=versions(),
           n_trials=n_trials, training_cv_accuracy=cv_acc,
           params=params, seconds=round(time.time() - t0, 1))
out["params_sha"] = sha16(json.dumps(params, sort_keys=True).encode())
save_json(out, os.path.join(OUT, "stage1_params.json"))
print(f"saved results/stage1_params.json  params_sha={out['params_sha']}", flush=True)
