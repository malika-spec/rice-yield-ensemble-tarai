"""
STAGE 6 — reviewer-requested tests (run once, after stage 5).
Run:    python stage6_reviewer_tests.py
Needs:  stage1_params.json (tuned values) and the dataset in this folder.
Output: results/stage6_before_after_tuning.csv   default vs tuned models
        results/stage6_calibration.csv           Brier score and calibration slope
        results/stage6_threshold_sensitivity.csv training-median threshold sensitivity
        results/stage6_meta.json                 fingerprints and versions

Nothing here changes model selection. The primary model stays [5,1,1] and the
stacking model stays LR without passthrough. Hold-out results are reported,
not used to choose anything.
"""
import time, warnings
warnings.filterwarnings("ignore")
from common import *
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

t0 = time.time()
D = load_data()
S1 = load_json(os.path.join(OUT, "stage1_params.json"))
assert S1["data_sha"] == D.data_sha, "dataset mismatch with stage 1"
P = S1["params"]
PV = "Weighted Soft Voting [5, 1, 1]"
PS = "Stacking LR-meta (no-passthrough)"
tuned_baselines, tuned_ens = candidates(P)
TUNED = {**tuned_baselines, **tuned_ens}

# ---------- (1) before and after tuning -----------------------------------
# "Before" = the same model families with library default hyperparameters.
DEFAULT = {
    "CatBoost (default)": lambda: make_pipe(CatBoostClassifier(
        verbose=0, random_seed=RANDOM_STATE, allow_writing_files=False, thread_count=1)),
    "XGBoost (default)": lambda: make_pipe(XGBClassifier(
        eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, verbosity=0, n_jobs=1)),
    "LightGBM (default)": lambda: make_pipe(LGBMClassifier(
        random_state=RANDOM_STATE, verbose=-1, n_jobs=1)),
    "Weighted Soft Voting [5,1,1] (default boosters)": lambda: make_pipe(VotingClassifier(
        estimators=[("cat", CatBoostClassifier(verbose=0, random_seed=RANDOM_STATE,
                                               allow_writing_files=False, thread_count=1)),
                    ("xgb", XGBClassifier(eval_metric="logloss", tree_method="hist",
                                          random_state=RANDOM_STATE, verbosity=0, n_jobs=1)),
                    ("lgb", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1))],
        voting="soft", weights=[5, 1, 1], n_jobs=1)),
}
pairs = [
    ("CatBoost", "CatBoost (default)", "CatBoost (tuned)"),
    ("XGBoost", "XGBoost (default)", "XGBoost (tuned)"),
    ("LightGBM", "LightGBM (default)", "LightGBM (tuned)"),
    ("Primary voting [5,1,1]", "Weighted Soft Voting [5,1,1] (default boosters)", PV),
]
ba_rows = []
for label, before_name, after_name in pairs:
    for stage, fac in [("before (default)", DEFAULT[before_name]), ("after (tuned)", TUNED[after_name])]:
        est = fac()
        scoring = ["accuracy", "roc_auc"] if is_soft(est) else ["accuracy"]
        r5 = cross_validate(est, D.X_tr, D.y_tr, cv=stratified_cv(), scoring=scoring, n_jobs=1)
        rr = cross_validate(fac(), D.X_tr, D.y_tr, cv=repeated_cv(), scoring="accuracy", n_jobs=1)
        m = fac().fit(D.X_tr, D.y_tr)
        pred = m.predict(D.X_te)
        ho_auc = roc_auc_score(D.y_te, m.predict_proba(D.X_te)[:, 1]) if is_soft(m) else float("nan")
        ba_rows.append(dict(
            model=label, stage=stage,
            cv5_acc=round(r5["test_accuracy"].mean(), 4), cv5_acc_sd=round(r5["test_accuracy"].std(), 4),
            cv5_auc=round(r5["test_roc_auc"].mean(), 4) if "test_roc_auc" in r5 else None,
            rep5x10_acc=round(rr["test_score"].mean(), 4), rep5x10_sd=round(rr["test_score"].std(), 4),
            holdout_acc=round(accuracy_score(D.y_te, pred), 4),
            holdout_auc=round(ho_auc, 4) if not np.isnan(ho_auc) else None))
        print(f"{label:<42} {stage:<18} rep {rr['test_score'].mean():.4f}  HO {accuracy_score(D.y_te, pred):.4f}", flush=True)
pd.DataFrame(ba_rows).to_csv(os.path.join(OUT, "stage6_before_after_tuning.csv"), index=False)

# ---------- (2) calibration on the hold-out probabilities ------------------
cal_rows = []
for name in ["CatBoost (tuned)", PV, PS]:
    m = TUNED[name]().fit(D.X_tr, D.y_tr)
    p = np.clip(m.predict_proba(D.X_te)[:, 1], 1e-6, 1 - 1e-6)
    y = np.asarray(D.y_te)
    brier = brier_score_loss(y, p)
    base = brier_score_loss(y, np.full(len(y), y.mean()))
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    cal_rows.append(dict(model=name, brier=round(brier, 4), brier_reference_base_rate=round(base, 4),
                         calibration_slope=round(float(lr.coef_[0][0]), 3),
                         calibration_intercept=round(float(lr.intercept_[0]), 3),
                         n_holdout=len(y), note="47 rows: indicative only"))
pd.DataFrame(cal_rows).to_csv(os.path.join(OUT, "stage6_calibration.csv"), index=False)

# ---------- (3) threshold sensitivity: training-median label ---------------
# Threshold computed on the 107 training rows only. Same split, same models,
# same primary configuration. Sensitivity analysis, not a new selection.
y_train_median = float(D.df[TARGET].iloc[D.X_tr.index].median())
y_alt = (D.df[TARGET] >= y_train_median).astype(int)
ytr_alt, yte_alt = y_alt.iloc[D.X_tr.index].values, y_alt.iloc[D.X_te.index].values
thr_rows = []
for name in ["CatBoost (tuned)", PV]:
    fac = TUNED[name]
    rr = cross_validate(fac(), D.X_tr, ytr_alt, cv=repeated_cv(), scoring="accuracy", n_jobs=1)
    m = fac().fit(D.X_tr, ytr_alt)
    pred = m.predict(D.X_te)
    auc = roc_auc_score(yte_alt, m.predict_proba(D.X_te)[:, 1])
    thr_rows.append(dict(model=name, threshold=round(y_train_median, 2),
                         train_high_share=round(float(ytr_alt.mean()), 3),
                         rep5x10_acc=round(rr["test_score"].mean(), 4),
                         rep5x10_sd=round(rr["test_score"].std(), 4),
                         holdout_acc=round(accuracy_score(yte_alt, pred), 4),
                         holdout_auc=round(auc, 4)))
    print(f"threshold sensitivity {name:<22} rep {rr['test_score'].mean():.4f}  HO {accuracy_score(yte_alt, pred):.4f}", flush=True)
pd.DataFrame(thr_rows).to_csv(os.path.join(OUT, "stage6_threshold_sensitivity.csv"), index=False)

save_json(dict(stage="stage6", data_sha=D.data_sha, params_sha=S1["params_sha"],
               versions=versions(), training_median_threshold=y_train_median,
               full_median_threshold=D.median, seconds=round(time.time() - t0, 1)),
          os.path.join(OUT, "stage6_meta.json"))
print("stage 6 done", flush=True)
