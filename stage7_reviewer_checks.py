"""
STAGE 7 — reviewer-requested checks (run once, after stage 6).
Run:    python stage7_reviewer_checks.py
Needs:  results/stage1_params.json, results/stage2_cv_summary.csv, results/stage3_nested_folds.csv
Writes ONLY new files:
        results/stage7_grouped_cv.csv        random vs year-grouped vs season-year-grouped CV (training rows)
        results/stage7_quantile_threshold.csv 40th and 60th percentile cut-points (hold-out reported)
        results/stage7_nested_default.csv    nested CV for default-parameter models, same outer folds as stage 3
        results/stage7_meta.json
No stage 1-6 file is read for writing and none is modified. Tuned hyperparameters are fixed.
The hold-out set is reported only; it is not used to choose anything.
"""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score
from common import *

t0 = time.time()
D = load_data()
S1 = load_json(os.path.join(OUT, "stage1_params.json"))
assert S1["data_sha"] == D.data_sha, "dataset mismatch with stage 1"
P = S1["params"]
baselines, ensembles = candidates(P)
ALL = {**baselines, **ensembles}
FOUR = ["CatBoost (tuned)", "Weighted Soft Voting [5, 1, 1]",
        "Stacking LR-meta (no-passthrough)", "Random Forest"]

def popsd(x): return float(np.std(x))  # population SD, as elsewhere in the study

# ---------- A. random vs grouped CV (training rows only) ----------------
Xtr, ytr = D.X_tr, D.y_tr
idx = Xtr.index.values
year_g = pd.Series(D.year[idx]).astype(str).values
sy_g = (pd.Series(D.year[idx]).astype(str) + "-" + pd.Series(D.season[idx]).astype(str)).values
s2 = pd.read_csv(os.path.join(OUT, "stage2_cv_summary.csv")).set_index("model")
grp_rows = []
for scheme, groups in [("year-grouped (11 groups, 5 folds)", year_g),
                       ("season-year-grouped (22 groups, 5 folds)", sy_g)]:
    for name in FOUR:
        r = cross_validate(ALL[name](), Xtr, ytr, cv=GroupKFold(n_splits=5), groups=groups,
                           scoring="accuracy", n_jobs=1)
        rand_acc = s2.loc[name, "rep5x10_acc"] if name in s2.index else np.nan
        rand_sd = s2.loc[name, "rep5x10_sd"] if name in s2.index else np.nan
        grp_rows.append(dict(model=name, scheme=scheme,
                             random_rep5x10_acc=round(float(rand_acc), 4),
                             random_rep5x10_sd=round(float(rand_sd), 4),
                             grouped_acc=round(float(r["test_score"].mean()), 4),
                             grouped_sd=round(popsd(r["test_score"]), 4)))
        print(f"{scheme:<42} {name:<36} random {rand_acc:.4f}  grouped {r['test_score'].mean():.4f}", flush=True)
pd.DataFrame(grp_rows).to_csv(os.path.join(OUT, "stage7_grouped_cv.csv"), index=False)

# ---------- B. quantile cut-points (hold-out reported, hyperparameters fixed) --
yield_all = D.df[TARGET].values
q_rows = []
for q in [0.40, 0.60]:
    cut = float(np.quantile(yield_all, q))
    y_all_q = (yield_all >= cut).astype(int)
    ytr_q = y_all_q[idx]
    yte_q = y_all_q[D.X_te.index.values]
    for name in FOUR:
        est = ALL[name]().fit(Xtr, ytr_q)
        pred = est.predict(D.X_te)
        acc = accuracy_score(yte_q, pred)
        auc = roc_auc_score(yte_q, est.predict_proba(D.X_te)[:, 1]) if hasattr(est, "predict_proba") else np.nan
        q_rows.append(dict(quantile=q, cut_value=round(cut, 3),
                           high_share_all=round(float(y_all_q.mean()), 3),
                           model=name, holdout_acc=round(acc, 4),
                           holdout_auc=round(float(auc), 4) if not np.isnan(auc) else None))
        print(f"q={q:.2f} cut={cut:.2f} {name:<36} hold-out acc {acc:.4f}", flush=True)

pd.DataFrame(q_rows).to_csv(os.path.join(OUT, "stage7_quantile_threshold.csv"), index=False)

# ---------- C. nested CV for default-parameter models (same outer folds as stage 3) ---
s3 = pd.read_csv(os.path.join(OUT, "stage3_nested_folds.csv"))
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)
folds = list(rskf.split(Xtr, ytr))
sizes = [(len(tr), len(te)) for tr, te in folds]
expected = list(zip(s3["n_train"].astype(int), s3["n_test"].astype(int)))
assert sizes == expected, f"outer fold sizes do not match stage 3: {sizes} vs {expected}"
print("outer fold sizes match stage 3 (10 folds)", flush=True)

DEFAULTS = ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "KNN"]
n_tr_mean = float(np.mean([a for a, _ in sizes])); n_te_mean = float(np.mean([b for _, b in sizes]))
ratio = n_te_mean / n_tr_mean
def corrected_t(diff):
    k = len(diff); m = diff.mean(); v = diff.var(ddof=1)
    t = m / np.sqrt((1.0 / k + ratio) * v)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))
def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(run, 1.0)
    return adj

cat_outer = s3["catboost_acc"].values.astype(float)
pri_outer = s3["primary_acc"].values.astype(float)
nest_rows = []
for name in DEFAULTS:
    accs, aucs = [], []
    for tr, te in folds:
        m = ALL[name]().fit(Xtr.iloc[tr], ytr.iloc[tr])
        accs.append(accuracy_score(ytr.iloc[te], m.predict(Xtr.iloc[te])))
        if hasattr(m, "predict_proba"):
            aucs.append(roc_auc_score(ytr.iloc[te], m.predict_proba(Xtr.iloc[te])[:, 1]))
    accs = np.array(accs)
    t_c, p_c = corrected_t(accs - cat_outer)
    t_p, p_p = corrected_t(accs - pri_outer)
    nest_rows.append(dict(model=name, nested_acc_mean=round(accs.mean(), 4),
                          nested_acc_sd=round(popsd(accs), 4),
                          nested_auc_mean=round(float(np.mean(aucs)), 4) if aucs else None,
                          diff_vs_catboost_nested=round(float((accs - cat_outer).mean()), 4), p_vs_catboost_raw=round(p_c, 4),
                          diff_vs_principal_nested=round(float((accs - pri_outer).mean()), 4), p_vs_principal_raw=round(p_p, 4)))
    print(f"nested {name:<22} acc {accs.mean():.4f} ± {popsd(accs):.4f}", flush=True)
nd = pd.DataFrame(nest_rows)
nd["p_vs_catboost_holm"] = np.round(holm(nd["p_vs_catboost_raw"].values), 4)
nd["p_vs_principal_holm"] = np.round(holm(nd["p_vs_principal_raw"].values), 4)
nd.to_csv(os.path.join(OUT, "stage7_nested_default.csv"), index=False)

save_json(dict(stage="stage7", data_sha=D.data_sha, params_sha=S1["params_sha"], versions=versions(),
               outer_fold_ratio=round(ratio, 4), seconds=round(time.time() - t0, 1)),
          os.path.join(OUT, "stage7_meta.json"))
print("stage 7 done", flush=True)
