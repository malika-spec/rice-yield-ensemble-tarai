"""
STAGE 3 — nested repeated CV (training data only).
Each outer fold re-runs Optuna (inner 5-fold CV) and the voting-weight
selection (inner repeated CV) using only its own training part, then scores
the selected model on the held-out outer fold.

Run one fold:      python stage3_nested.py --fold 0      (folds 0..9)
Run all folds:     python stage3_nested.py
Combine folds:     python stage3_nested.py --summarize
Parallel: start several terminals with different --fold values. Each fold
writes its own file, so runs do not interfere, and results match a
sequential run because every step is seeded.
Output: results/stage3/fold_XX.json, then results/stage3_nested_summary.json
        and results/stage3_nested_folds.csv
Time:   about 2-3 minutes per fold on one core (about 25-40 min in total).
"""
import sys, time, argparse
from common import *
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, default=None)
ap.add_argument("--summarize", action="store_true")
args = ap.parse_args()

FOLD_DIR = os.path.join(OUT, "stage3")
os.makedirs(FOLD_DIR, exist_ok=True)
D = load_data()
N_OUTER = CV_FOLDS * OUTER_REPEATS

def fold_path(i):
    rep, f = divmod(i, CV_FOLDS)
    return os.path.join(FOLD_DIR, f"fold_r{rep+1}_f{f+1}.json")

if args.summarize:
    recs = []
    for i in range(N_OUTER):
        p = fold_path(i)
        assert os.path.exists(p), f"missing fold {i}: {p}"
        r = load_json(p)
        assert r["data_sha"] == D.data_sha, f"fold {i} came from a different dataset"
        recs.append(r)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "stage3_nested_folds.csv"), index=False)
    summary = dict(stage="stage3_nested", data_sha=D.data_sha, versions=versions(),
                   n_outer_folds=len(df),
                   primary_acc_mean=df.primary_acc.mean(), primary_acc_sd=df.primary_acc.std(ddof=0),
                   primary_auc_mean=df.primary_auc.mean(),
                   catboost_acc_mean=df.catboost_acc.mean(), catboost_acc_sd=df.catboost_acc.std(ddof=0),
                   catboost_auc_mean=df.catboost_auc.mean(),
                   primary_weights_chosen=df.primary_weights.value_counts().to_dict(),
                   inner_selection="repeated 5-fold x INNER_REPEATS on outer training part")
    save_json(summary, os.path.join(OUT, "stage3_nested_summary.json"))
    print(df.round(4).to_string(index=False), flush=True)
    print(f"primary nested accuracy {summary['primary_acc_mean']:.4f} +/- {summary['primary_acc_sd']:.4f}")
    print(f"CatBoost nested accuracy {summary['catboost_acc_mean']:.4f} +/- {summary['catboost_acc_sd']:.4f}")
    sys.exit(0)

folds = [args.fold] if args.fold is not None else list(range(N_OUTER))
outer = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)
splits = list(outer.split(D.X_tr, D.y_tr))

for i in folds:
    if os.path.exists(fold_path(i)):
        print(f"fold {i}: already done, skipping", flush=True); continue
    t0 = time.time()
    tr, te = splits[i]
    Xa, ya = D.X_tr.iloc[tr], D.y_tr.iloc[tr]
    Xb, yb = D.X_tr.iloc[te], D.y_tr.iloc[te]

    # (a) Optuna inside this outer training fold only
    P = {}
    for name in ["CatBoost", "XGBoost", "LightGBM"]:
        P[name], _, _ = tune_on(name, Xa, ya)

    # (b) voting-weight selection inside this outer training fold only
    inner = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=INNER_REPEATS, random_state=RANDOM_STATE)
    scores = {}
    for w in WEIGHT_CONFIGS:
        s = cross_validate(voting_pipe(w, P), Xa, ya, cv=inner, scoring="accuracy", n_jobs=1)
        scores[str(w)] = float(s["test_score"].mean())
    chosen = max(WEIGHT_CONFIGS, key=lambda w: (round(scores[str(w)], 10), -WEIGHT_CONFIGS.index(w)))

    # (c) score the selected model and the CatBoost baseline on the outer fold
    m_vote = voting_pipe(chosen, P).fit(Xa, ya)
    m_cat = make_pipe(build_cat(P["CatBoost"])).fit(Xa, ya)
    pv, pc = m_vote.predict_proba(Xb)[:, 1], m_cat.predict_proba(Xb)[:, 1]
    rec = dict(stage="stage3_nested", fold_index=i,
               repeat=i // CV_FOLDS + 1, fold=i % CV_FOLDS + 1,
               n_train=len(tr), n_test=len(te),
               data_sha=D.data_sha, versions=versions(),
               tuned_params=P, inner_weight_scores=scores,
               primary_weights=str(chosen),
               primary_acc=accuracy_score(yb, m_vote.predict(Xb)),
               primary_auc=roc_auc_score(yb, pv),
               catboost_acc=accuracy_score(yb, m_cat.predict(Xb)),
               catboost_auc=roc_auc_score(yb, pc),
               seconds=round(time.time() - t0, 1))
    rec["params_sha_context"] = sha16(json.dumps(P, sort_keys=True).encode())
    save_json(rec, fold_path(i))
    print(f"fold {i}: weights {chosen}  primary {rec['primary_acc']:.3f}  "
          f"CatBoost {rec['catboost_acc']:.3f}  ({rec['seconds']}s)", flush=True)
