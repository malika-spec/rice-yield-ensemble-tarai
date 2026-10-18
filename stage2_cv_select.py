"""
STAGE 2 — repeated 5x10 CV on the training rows, then model selection.
Run:    python stage2_cv_select.py
Needs:  results/stage1_params.json
Output: results/stage2_cv_summary.csv, results/stage2_folds.json,
        results/stage2_selection.json
Time:   about 20-35 minutes on a single core.
"""
import time
from common import *

t0 = time.time()
D = load_data()
S1 = load_json(os.path.join(OUT, "stage1_params.json"))
assert S1["data_sha"] == D.data_sha, "stage1 was run on a different dataset"
P = S1["params"]
baselines, ensembles = candidates(P)
ALL = {**baselines, **ensembles}

def cv_eval(name, factory):
    est = factory()
    scoring = ["accuracy", "roc_auc"] if is_soft(est) else ["accuracy"]
    r5 = cross_validate(est, D.X_tr, D.y_tr, cv=stratified_cv(), scoring=scoring,
                        return_train_score=True, n_jobs=1)
    rr = cross_validate(factory(), D.X_tr, D.y_tr, cv=repeated_cv(),
                        scoring="accuracy", n_jobs=1)
    has_auc = "test_roc_auc" in r5
    return dict(model=name,
                cv5_acc_mean=r5["test_accuracy"].mean(), cv5_acc_sd=r5["test_accuracy"].std(),
                cv5_auc_mean=r5["test_roc_auc"].mean() if has_auc else float("nan"),
                cv5_auc_sd=r5["test_roc_auc"].std() if has_auc else float("nan"),
                train_acc=r5["train_accuracy"].mean(),
                rep_acc_mean=rr["test_score"].mean(), rep_acc_sd=rr["test_score"].std(),
                rep_folds=[float(x) for x in rr["test_score"]])

results = {}
for name, f in ALL.items():
    results[name] = cv_eval(name, f)
    r = results[name]
    print(f"{name:<44} 5-fold {r['cv5_acc_mean']:.4f}  5x10 {r['rep_acc_mean']:.4f}", flush=True)

# Pre-specified selection rule: highest mean repeated 5x10 CV accuracy on training
# data; ties broken by 5-fold CV AUC. The hold-out set plays no part.
primary_voting = max([wname(w) for w in WEIGHT_CONFIGS], key=lambda n: selection_key(results[n]))
primary_stack = max([n for n in ensembles if n.startswith("Stacking")],
                    key=lambda n: selection_key(results[n]))

rows = [dict(model=n, cv5_acc=round(r["cv5_acc_mean"], 4), cv5_acc_sd=round(r["cv5_acc_sd"], 4),
             cv5_auc=round(r["cv5_auc_mean"], 4) if not np.isnan(r["cv5_auc_mean"]) else None,
             rep5x10_acc=round(r["rep_acc_mean"], 4), rep5x10_sd=round(r["rep_acc_sd"], 4),
             train_acc=round(r["train_acc"], 4))
        for n, r in results.items()]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage2_cv_summary.csv"), index=False)
save_json({n: r["rep_folds"] for n, r in results.items()}, os.path.join(OUT, "stage2_folds.json"))
save_json(dict(stage="stage2_cv_select", data_sha=D.data_sha, params_sha=S1["params_sha"],
               versions=versions(), rule="max mean repeated 5x10 CV accuracy; tie -> 5-fold AUC",
               primary_voting=primary_voting, primary_stacking=primary_stack,
               seconds=round(time.time() - t0, 1)),
          os.path.join(OUT, "stage2_selection.json"))
print(f"primary voting: {primary_voting}", flush=True)
print(f"primary stacking: {primary_stack}", flush=True)
