"""
common.py — shared definitions for the staged pipeline.
Every stage imports this file, so data, split, search spaces and model
definitions are identical across stages. Each stage writes a fingerprint
(SHA-256 of the dataset bytes, the Optuna parameters, and library versions)
so check_consistency.py can confirm that all outputs came from one run.
"""
import os, json, hashlib, platform, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import sklearn, catboost, xgboost, lightgbm, shap, scipy
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (VotingClassifier, StackingClassifier,
                              RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------
# Configuration (manuscript values; override only via environment)
# ---------------------------------------------------------------------
DATA_PATH     = os.environ.get("DATA_PATH", "rice_yield_dataset.csv")
TARGET        = "Yield (quintal/hectre)"
META_COLS     = ["year", "season"]
OUT           = "results"
RANDOM_STATE  = 42
TEST_SIZE     = 0.30
CV_FOLDS      = 5
N_REPEATS     = int(os.environ.get("N_REPEATS", 10))
N_TRIALS      = int(os.environ.get("N_TRIALS", 50))
N_BOOT        = int(os.environ.get("N_BOOT", 1000))
DPI           = int(os.environ.get("DPI", 600))
SHAP_NSAMPLES = int(os.environ.get("SHAP_NSAMPLES", 300))
OUTER_REPEATS = int(os.environ.get("OUTER_REPEATS", 2))
INNER_REPEATS = int(os.environ.get("INNER_REPEATS", 1))

os.makedirs(OUT, exist_ok=True)

def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]

def versions():
    return dict(python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, optuna=optuna.__version__,
                catboost=catboost.__version__, xgboost=xgboost.__version__,
                lightgbm=lightgbm.__version__, shap=shap.__version__)

def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

def load_json(path):
    with open(path) as fh:
        return json.load(fh)

# ---------------------------------------------------------------------
# Data, split and fingerprint
# ---------------------------------------------------------------------
class Data:
    pass

def load_data():
    D = Data()
    with open(DATA_PATH, "rb") as fh:
        raw = fh.read()
    D.data_sha = sha16(raw)
    df = pd.read_csv(DATA_PATH)
    assert len(df) == 154, "expected 154 rows"
    assert set(META_COLS) <= set(df.columns)
    assert df[["year", "season"]].drop_duplicates().shape[0] == 22
    D.df = df
    D.median = float(df[TARGET].median())
    D.y_all = (df[TARGET] >= D.median).astype(int)
    D.X_all = df.drop(columns=[TARGET] + META_COLS)
    assert D.X_all.shape[1] == 18
    D.features = list(D.X_all.columns)
    D.year = df["year"].values
    D.season = df["season"].values
    D.X_tr, D.X_te, D.y_tr, D.y_te = train_test_split(
        D.X_all, D.y_all, test_size=TEST_SIZE, stratify=D.y_all, random_state=RANDOM_STATE)
    assert len(D.X_tr) == 107 and len(D.X_te) == 47
    assert D.y_tr.value_counts().to_dict() == {1: 54, 0: 53}
    assert D.y_te.value_counts().to_dict() == {0: 24, 1: 23}
    return D

# ---------------------------------------------------------------------
# Pipelines, CV objects, search spaces
# ---------------------------------------------------------------------
def make_pipe(est):
    return Pipeline([("imp", SimpleImputer(strategy="mean")),
                     ("sc", StandardScaler()), ("clf", est)])

def stratified_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def repeated_cv(repeats=None):
    return RepeatedStratifiedKFold(n_splits=CV_FOLDS,
                                   n_repeats=repeats or N_REPEATS,
                                   random_state=RANDOM_STATE)

def space_cat(t):
    return dict(depth=t.suggest_int("depth", 4, 10),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.3),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 10.0),
                iterations=t.suggest_int("iterations", 100, 1000),
                border_count=t.suggest_int("border_count", 32, 255),
                random_strength=t.suggest_float("random_strength", 1, 20),
                bagging_temperature=t.suggest_float("bagging_temperature", 0, 1))

def space_xgb(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 100, 1500),
                max_depth=t.suggest_int("max_depth", 3, 10),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.3),
                reg_lambda=t.suggest_float("reg_lambda", 0.001, 2.0),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                gamma=t.suggest_float("gamma", 0, 5))

def space_lgb(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 100, 1500),
                max_depth=t.suggest_int("max_depth", 3, 10),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.3),
                num_leaves=t.suggest_int("num_leaves", 20, 100),
                reg_lambda=t.suggest_float("reg_lambda", 0.001, 2.0),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0))

def build_cat(p): return CatBoostClassifier(**p, verbose=0, random_seed=RANDOM_STATE,
                                            allow_writing_files=False, thread_count=1)
def build_xgb(p): return XGBClassifier(**p, eval_metric="logloss", tree_method="hist",
                                       random_state=RANDOM_STATE, verbosity=0, n_jobs=1)
def build_lgb(p): return LGBMClassifier(**p, random_state=RANDOM_STATE, verbose=-1, n_jobs=1)

SPACES = {"CatBoost": (space_cat, build_cat),
          "XGBoost": (space_xgb, build_xgb),
          "LightGBM": (space_lgb, build_lgb)}

def tune_on(name, Xd, yd):
    """Optuna TPE study (seed 42). Objective: mean accuracy of 5-fold stratified
    CV on (Xd, yd). Called with training rows for the main tuning, and with each
    outer training fold for nested CV. The 47 hold-out rows are never passed in."""
    space, build = SPACES[name]
    cv = stratified_cv()
    def objective(trial):
        s = cross_validate(make_pipe(build(space(trial))), Xd, yd, cv=cv,
                           scoring="accuracy", n_jobs=1)
        return s["test_score"].mean()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, float(study.best_value), len(study.trials)

# ---------------------------------------------------------------------
# Candidate models (built from a parameter dictionary)
# ---------------------------------------------------------------------
WEIGHT_CONFIGS = [[1,1,1],[2,1,1],[3,1,1],[4,1,1],[5,1,1],
                  [3,2,1],[3,3,1],[4,2,1],[2,2,1]]

def wname(w): return f"Weighted Soft Voting {w}"

def base3(P):
    return [("cat", build_cat(P["CatBoost"])), ("xgb", build_xgb(P["XGBoost"])),
            ("lgb", build_lgb(P["LightGBM"]))]

def base4(P):
    return base3(P) + [("rf", RandomForestClassifier(n_estimators=100,
                                                     random_state=RANDOM_STATE, n_jobs=1))]

def voting_pipe(w, P):
    return make_pipe(VotingClassifier(estimators=base3(P), voting="soft",
                                      weights=w, n_jobs=1))

def candidates(P):
    """All models in the manuscript, built from parameter dict P."""
    baselines = {
        "CatBoost (tuned)": lambda: make_pipe(build_cat(P["CatBoost"])),
        "XGBoost (tuned)":  lambda: make_pipe(build_xgb(P["XGBoost"])),
        "LightGBM (tuned)": lambda: make_pipe(build_lgb(P["LightGBM"])),
        "Random Forest":    lambda: make_pipe(RandomForestClassifier(
                                n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)),
        "Logistic Regression": lambda: make_pipe(LogisticRegression(
                                max_iter=1000, random_state=RANDOM_STATE)),
        "Gradient Boosting": lambda: make_pipe(GradientBoostingClassifier(
                                random_state=RANDOM_STATE)),
        "SVM": lambda: make_pipe(SVC(probability=True, random_state=RANDOM_STATE)),
        "KNN": lambda: make_pipe(KNeighborsClassifier()),
    }
    ensembles = {wname(w): (lambda w=w: voting_pipe(w, P)) for w in WEIGHT_CONFIGS}
    ensembles["Weighted Soft Voting [3,2,1,1] (4 base)"] = lambda: make_pipe(
        VotingClassifier(estimators=base4(P), voting="soft", weights=[3,2,1,1], n_jobs=1))
    ensembles["Hard Voting (3 base)"] = lambda: make_pipe(
        VotingClassifier(estimators=base3(P), voting="hard", n_jobs=1))
    for passthrough in [True, False]:
        tag = "passthrough" if passthrough else "no-passthrough"
        ensembles[f"Stacking LR-meta ({tag})"] = (lambda p=passthrough: make_pipe(StackingClassifier(
            estimators=base3(P),
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            passthrough=p, cv=stratified_cv(), n_jobs=1)))
        ensembles[f"Stacking RF-meta ({tag})"] = (lambda p=passthrough: make_pipe(StackingClassifier(
            estimators=base3(P),
            final_estimator=RandomForestClassifier(n_estimators=100,
                                                   random_state=RANDOM_STATE, n_jobs=1),
            passthrough=p, cv=stratified_cv(), n_jobs=1)))
    return baselines, ensembles

def is_soft(est):
    return hasattr(est, "predict_proba") and getattr(est, "voting", "soft") != "hard"

def selection_key(res):
    auc = res["cv5_auc_mean"] if not np.isnan(res["cv5_auc_mean"]) else -1
    return (round(res["rep_acc_mean"], 10), auc)
