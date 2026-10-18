"""
STAGE 5 — SHAP, learning curves, figures (600 DPI).
Run:    python stage5_shap_figures.py
Needs:  stage1_params.json, stage2_selection.json, stage2_cv_summary.csv,
        stage4_holdout.csv, stage4_holdout_probs.npz, stage4_temporal.csv
Output: results/stage5_shap.csv, results/stage5_learning_curves.csv,
        results/figures/*.png
"""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from common import *
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve

t0 = time.time()
D = load_data()
S1 = load_json(os.path.join(OUT, "stage1_params.json"))
S2 = load_json(os.path.join(OUT, "stage2_selection.json"))
S4 = load_json(os.path.join(OUT, "stage4_meta.json"))
assert S1["data_sha"] == D.data_sha == S2["data_sha"] == S4["data_sha"], "dataset mismatch"
assert S4["primary_voting"] == S2["primary_voting"], "selection changed between stages"
P = S1["params"]; PV, PS = S2["primary_voting"], S2["primary_stacking"]
baselines, ensembles = candidates(P)
ALL = {**baselines, **ensembles}
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)

# ---------- SHAP ----------------------------------------------------------
Xte = pd.DataFrame(D.X_te.values, columns=D.features)
Xtr = pd.DataFrame(D.X_tr.values, columns=D.features)
cat = make_pipe(build_cat(P["CatBoost"])).fit(D.X_tr, D.y_tr)
sv_cat = shap.TreeExplainer(cat.named_steps["clf"]).shap_values(cat[:-1].transform(Xte))
sv_cat = sv_cat[1] if isinstance(sv_cat, list) else sv_cat
prim = ALL[PV]().fit(D.X_tr, D.y_tr)
bg = shap.sample(Xtr, 50, random_state=RANDOM_STATE)
np.random.seed(RANDOM_STATE)  # fixed seed: KernelSHAP sampling is otherwise unreproducible
kex = shap.KernelExplainer(lambda Z: prim.predict_proba(pd.DataFrame(Z, columns=D.features))[:, 1], bg)
sv_ens = kex.shap_values(Xte, nsamples=SHAP_NSAMPLES, silent=True)
shap_df = pd.DataFrame({"CatBoost (TreeSHAP)": np.abs(sv_cat).mean(0),
                        f"{PV} (KernelSHAP)": np.abs(sv_ens).mean(0)}, index=D.features)
shap_df.round(4).to_csv(os.path.join(OUT, "stage5_shap.csv"))

# Direction check: Spearman correlation between a feature's value and its SHAP value
# on the hold-out rows. A positive rho means higher values are associated with a higher
# predicted probability of the High class. Association only, not causal.
from scipy.stats import spearmanr
dir_rows = []
for f in ["humidity", "rain_1h", "Area (Ha)", "pressure", "wind_speed"]:
    j = D.features.index(f)
    for name, sv in [("CatBoost (TreeSHAP)", sv_cat), (f"{PV} (KernelSHAP)", sv_ens)]:
        rho, p = spearmanr(D.X_te.values[:, j], sv[:, j])
        dir_rows.append(dict(feature=f, model=name, spearman_rho=round(float(rho), 3),
                             p_value=round(float(p), 4),
                             mean_abs_shap=round(float(np.abs(sv[:, j]).mean()), 4)))
pd.DataFrame(dir_rows).to_csv(os.path.join(OUT, "stage5_shap_direction.csv"), index=False)
ax = shap_df.sort_values(shap_df.columns[1]).plot.barh(figsize=(9, 7), alpha=.85)
ax.set_xlabel("Mean |SHAP value| (hold-out rows)")
ax.set_title("Feature importance, primary ensemble", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_importance.png"), dpi=DPI); plt.close()

# ---------- learning curves (training data only) -------------------------
SIZES = np.linspace(0.2, 1.0, 8)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5)); lc = []
for axis, name in zip(axes, [PV, wname([1,1,1]), PS]):
    ts, trs, cvs = learning_curve(ALL[name](), D.X_tr, D.y_tr, train_sizes=SIZES,
                                  cv=stratified_cv(), scoring="accuracy", n_jobs=1)
    for i, n in enumerate(ts):
        lc.append(dict(model=name, n_train=int(n), train_acc=round(trs[i].mean(), 4),
                       train_sd=round(trs[i].std(), 4), cv_acc=round(cvs[i].mean(), 4),
                       cv_sd=round(cvs[i].std(), 4)))
    axis.plot(ts, trs.mean(1), "o-", label="Training")
    axis.fill_between(ts, trs.mean(1)-trs.std(1), trs.mean(1)+trs.std(1), alpha=.15)
    axis.plot(ts, cvs.mean(1), "s--", label="Validation (5-fold CV)")
    axis.fill_between(ts, cvs.mean(1)-cvs.std(1), cvs.mean(1)+cvs.std(1), alpha=.15)
    axis.set_title(name, fontweight="bold"); axis.set_xlabel("Training samples")
    axis.set_ylabel("Accuracy"); axis.set_ylim(0.4, 1.02); axis.grid(alpha=.25); axis.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "learning_curves.png"), dpi=DPI); plt.close()
pd.DataFrame(lc).to_csv(os.path.join(OUT, "stage5_learning_curves.csv"), index=False)

# ---------- ROC figures (hold-out) --------------------------------------
probs = np.load(os.path.join(OUT, "stage4_holdout_probs.npz"))
def roc_fig(names, title, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    colours = plt.cm.tab20(np.linspace(0, 1, max(len(names), 2)))
    for i, n in enumerate(names):
        key = n.replace(" ", "_")
        if key not in probs.files: continue
        fpr, tpr, _ = roc_curve(D.y_te, probs[key])
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, lw=2, ls="--" if "Stacking" in n else "-", color=colours[i],
                label=f"{n} (AUC={roc_auc_score(D.y_te, probs[key]):.3f})")
    ax.plot([0,1],[0,1],"k:", lw=1, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(os.path.join(FIG, fname), dpi=DPI); plt.close()
roc_fig(list(baselines), "Hold-out ROC: baseline models", "roc_baselines.png")
roc_fig([n for n in ensembles if "Hard" not in n],
        "Hold-out ROC: ensemble models (dashed = stacking)", "roc_ensembles.png")

# ---------- weight sensitivity and temporal figures ----------------------
cvs_df = pd.read_csv(os.path.join(OUT, "stage2_cv_summary.csv"))
wdf = cvs_df[cvs_df.model.str.startswith("Weighted Soft Voting [")]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(wdf.model.str.replace("Weighted Soft Voting ", ""), wdf.rep5x10_acc,
        color=["#d62728" if m == PV else "#1f77b4" for m in wdf.model], alpha=.85)
ax.set_xlim(0.80, 0.92); ax.set_xlabel("Repeated 5x10 CV accuracy (training data)")
ax.set_title("Voting-weight sensitivity (selected on training data)", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "weight_sensitivity.png"), dpi=DPI); plt.close()

tdf = pd.read_csv(os.path.join(OUT, "stage4_temporal.csv"))
fig, ax = plt.subplots(figsize=(10, 6))
labels = tdf.model.str.slice(0, 28) + " | " + tdf.scheme.str.slice(0, 40)
ax.barh(labels, tdf.test_accuracy, color="#2ca02c", alpha=.8)
ax.axvline(0.5, color="k", ls="--", lw=1); ax.set_xlabel("Test accuracy")
ax.set_title("Temporal validation (fixed configurations)", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "temporal_validation.png"), dpi=DPI); plt.close()

save_json(dict(stage="stage5", data_sha=D.data_sha, versions=versions(),
               dpi=DPI, seconds=round(time.time() - t0, 1)), os.path.join(OUT, "stage5_meta.json"))
print("stage 5 done", flush=True)
