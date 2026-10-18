"""
STAGE 4 — hold-out evaluation (once), statistics, temporal validation,
outlier sensitivity.
Run:    python stage4_holdout_stats.py
Needs:  stage1_params.json, stage2_selection.json, stage2_folds.json,
        stage3_nested_folds.csv (run stage3 --summarize first)
Output: results/stage4_*.csv
Time:   about 5-10 minutes.
"""
import math, time
from common import *
from scipy import stats
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

t0 = time.time()
D = load_data()
S1 = load_json(os.path.join(OUT, "stage1_params.json"))
S2 = load_json(os.path.join(OUT, "stage2_selection.json"))
S3 = load_json(os.path.join(OUT, "stage3_nested_summary.json"))
assert S1["data_sha"] == D.data_sha == S2["data_sha"] == S3["data_sha"], "dataset mismatch between stages"
assert S2["params_sha"] == S1["params_sha"], "selection was made with different parameters"
P = S1["params"]
PV, PS = S2["primary_voting"], S2["primary_stacking"]
baselines, ensembles = candidates(P)
ALL = {**baselines, **ensembles}
folds2 = load_json(os.path.join(OUT, "stage2_folds.json"))
nested = pd.read_csv(os.path.join(OUT, "stage3_nested_folds.csv"))

# ---------- (A) hold-out: each model fitted once on all 107 rows -------
def wilson(p, n, z=1.96):
    d = 1 + z**2/n; c = (p + z**2/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))/d
    return max(0, c-h), min(1, c+h)

def boot(yt, yp, ypred, metric):
    rng = np.random.RandomState(RANDOM_STATE); vals = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(yt), len(yt))
        if len(np.unique(yt[i])) < 2: continue
        if metric == "auc": vals.append(roc_auc_score(yt[i], yp[i]))
        elif metric == "f1": vals.append(f1_score(yt[i], ypred[i], zero_division=0))
        elif metric == "precision": vals.append(precision_score(yt[i], ypred[i], zero_division=0))
        elif metric == "recall": vals.append(recall_score(yt[i], ypred[i], zero_division=0))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)

yt = np.asarray(D.y_te)
hold_rows, probs = [], {}
for name, fac in ALL.items():
    est = fac().fit(D.X_tr, D.y_tr)
    pred = est.predict(D.X_te)
    acc = accuracy_score(D.y_te, pred); lo, hi = wilson(acc, len(D.y_te))
    row = dict(model=name, ho_acc=round(acc, 4), wilson_lo=round(lo, 4), wilson_hi=round(hi, 4))
    if hasattr(est, "predict_proba") and getattr(est, "voting", "soft") != "hard":
        prob = est.predict_proba(D.X_te)[:, 1]; probs[name] = prob
        row["ho_auc"] = round(roc_auc_score(D.y_te, prob), 4)
        for m in ["auc", "f1", "precision", "recall"]:
            lo_b, hi_b = boot(yt, prob, np.asarray(pred), m)
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = lo_b, hi_b
    tn, fp, fn, tp = confusion_matrix(D.y_te, pred).ravel()
    row.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    hold_rows.append(row)
    print(f"{name:<44} hold-out {acc:.4f}", flush=True)
pd.DataFrame(hold_rows).to_csv(os.path.join(OUT, "stage4_holdout.csv"), index=False)
np.savez(os.path.join(OUT, "stage4_holdout_probs.npz"), **{k.replace(" ", "_"): v for k, v in probs.items()})

# ---------- (B) statistics: Nadeau-Bengio corrected resampled t-test ----
def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def corrected_t(diff, nt):
    diff = np.asarray(diff, float); k = len(diff); m, v = diff.mean(), diff.var(ddof=1)
    if v == 0: return m, float("nan"), (1.0 if m == 0 else 0.0), (m, m)
    se = math.sqrt((1/k + nt) * v); t = m/se
    tc = stats.t.ppf(0.975, df=k-1)
    return m, t, 2*stats.t.sf(abs(t), df=k-1), (m - tc*se, m + tc*se)

def rank_biserial(d):
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0: return 0.0
    r = stats.rankdata(np.abs(d)); wp, wn = r[d > 0].sum(), r[d < 0].sum()
    return (wp - wn) / (wp + wn)

NT = (1/CV_FOLDS) / (1 - 1/CV_FOLDS)   # test/train ratio of a 5-fold split = 0.25
ref = np.asarray(folds2[PV])
rows = []
for name, f in folds2.items():
    if name == PV: continue
    f = np.asarray(f)
    m, t, p, ci = corrected_t(ref - f, NT)
    rows.append(dict(analysis="repeated 5x10 CV (training)", reference=PV, comparison=name,
                     mean_diff=round(m, 4), ci95_low=round(ci[0], 4), ci95_high=round(ci[1], 4),
                     t_corrected=round(t, 3) if not np.isnan(t) else None,
                     p_corrected=round(p, 4), rank_biserial=round(rank_biserial(ref - f), 3)))
st = pd.DataFrame(rows); st["p_holm"] = np.round(holm(st.p_corrected.values), 4)

n_ref = nested.primary_acc.values; n_cat = nested.catboost_acc.values
m, t, p, ci = corrected_t(n_ref - n_cat, NT)
nested_row = pd.DataFrame([dict(analysis="nested repeated CV (training)", reference="Primary voting (nested)",
                                comparison="CatBoost (nested)", mean_diff=round(m, 4),
                                ci95_low=round(ci[0], 4), ci95_high=round(ci[1], 4),
                                t_corrected=round(t, 3) if not np.isnan(t) else None,
                                p_corrected=round(p, 4),
                                rank_biserial=round(rank_biserial(n_ref - n_cat), 3),
                                p_holm=round(p, 4))])
st = pd.concat([st, nested_row], ignore_index=True)
st.to_csv(os.path.join(OUT, "stage4_stats.csv"), index=False)

# optimism: internal estimate minus nested estimate, primary model
internal = folds2[PV]
optimism = dict(primary_internal_5x10=float(np.mean(internal)),
                primary_nested=float(n_ref.mean()),
                difference=float(np.mean(internal) - n_ref.mean()))
save_json(optimism, os.path.join(OUT, "stage4_optimism.json"))

# ---------- (C) temporal validation: reviewer-recommended schemes -------
TEMP_MODELS = {"CatBoost (tuned)": baselines["CatBoost (tuned)"], PV: ensembles[PV],
               PS: ensembles[PS], "Random Forest": baselines["Random Forest"]}
def split_eval(name, fac, tr, te, scheme):
    est = fac().fit(D.X_all.iloc[tr], D.y_all.values[tr])
    pred = est.predict(D.X_all.iloc[te])
    prob = est.predict_proba(D.X_all.iloc[te])[:, 1] if is_soft(est) else None
    auc = roc_auc_score(D.y_all.values[te], prob) if prob is not None and len(np.unique(D.y_all.values[te])) == 2 else float("nan")
    return dict(model=name, scheme=scheme, n_train=len(tr), n_test=len(te),
                test_accuracy=round(accuracy_score(D.y_all.values[te], pred), 4),
                test_auc=round(auc, 4) if not np.isnan(auc) else None,
                share_predicted_high=round(float(np.mean(pred)), 3),
                share_true_high=round(float(np.mean(D.y_all.values[te])), 3))
temp = []
for name, fac in TEMP_MODELS.items():
    for a, b in [("Kharif", "Zaid"), ("Zaid", "Kharif")]:
        temp.append(split_eval(name, fac, np.where(D.season == a)[0], np.where(D.season == b)[0],
                               f"leave-one-season-out: train {a} -> test {b}"))
    temp.append(split_eval(name, fac, np.where(D.year <= 2018)[0], np.where(D.year > 2018)[0],
                           "forward: train 2013-2018 -> test 2019-2023"))
pd.DataFrame(temp).to_csv(os.path.join(OUT, "stage4_temporal.csv"), index=False)

# ---------- (D) outlier sensitivity: IQR winsorisation (training-fitted) -
q1, q3 = D.X_tr.quantile(0.25), D.X_tr.quantile(0.75)
lo_f, hi_f = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
Xw_tr = D.X_tr.clip(lower=lo_f, upper=hi_f, axis=1); Xw_te = D.X_te.clip(lower=lo_f, upper=hi_f, axis=1)
n_capped = {c: int(((D.X_tr[c] < lo_f[c]) | (D.X_tr[c] > hi_f[c])).sum()) for c in D.features}
sens = []
for name in ["CatBoost (tuned)", PV, PS]:
    rr = cross_validate(ALL[name](), Xw_tr, D.y_tr, cv=repeated_cv(), scoring="accuracy", n_jobs=1)
    est = ALL[name]().fit(Xw_tr, D.y_tr)
    sens.append(dict(model=name, rep_acc_original=round(float(np.mean(folds2[name])), 4),
                     rep_acc_winsorised=round(float(rr["test_score"].mean()), 4),
                     holdout_acc_winsorised=round(accuracy_score(D.y_te, est.predict(Xw_te)), 4)))
pd.DataFrame(sens).to_csv(os.path.join(OUT, "stage4_outlier_sensitivity.csv"), index=False)
save_json(n_capped, os.path.join(OUT, "stage4_outlier_counts.json"))

# ---------- (E) univariate tests on training data (High vs Low) ---------
tr = D.X_tr.copy(); tr["y"] = D.y_tr.values
mw = []
for f_ in D.features:
    h = tr.loc[tr.y == 1, f_].dropna(); l = tr.loc[tr.y == 0, f_].dropna()
    U, p = stats.mannwhitneyu(h, l, alternative="two-sided")
    mw.append(dict(feature=f_, U=U, p=round(p, 4), rank_biserial=round(1 - 2*U/(len(h)*len(l)), 3)))
mwdf = pd.DataFrame(mw); mwdf["p_holm"] = np.round(holm(mwdf.p.values), 4)
mwdf.to_csv(os.path.join(OUT, "stage4_mannwhitney.csv"), index=False)

save_json(dict(stage="stage4", data_sha=D.data_sha, params_sha=S1["params_sha"],
               versions=versions(), primary_voting=PV, primary_stacking=PS,
               seconds=round(time.time() - t0, 1)),
          os.path.join(OUT, "stage4_meta.json"))
print("stage 4 done", flush=True)
