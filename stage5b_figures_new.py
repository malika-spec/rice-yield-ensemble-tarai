"""
STAGE 5b — figures regenerated from saved outputs (no model is refitted).
Run after stage 5:  python stage5b_figures.py
Reads:  results/stage4_holdout.csv, results/stage4_holdout_probs.npz,
        results/stage2_cv_summary.csv, results/stage4_temporal.csv,
        results/stage5_shap.csv
Writes: results/figures/roc_individual/*.png   one ROC figure per model
        results/figures/weight_sensitivity.png (nine 3-base configurations only)
        results/figures/temporal_validation.png (readable labels)
        results/figures/shap_importance.png     (two panels, separate scales)
        results/figures/roc_ensembles.png and roc_baselines.png are replaced
        by the individual figures; the combined files are removed.
All figures use DPI from common.py (600).
"""
import os, re, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from common import load_data, DPI, OUT, sha16

D = load_data()
FIG = os.path.join(OUT, "figures")
IND = os.path.join(FIG, "roc_individual")
os.makedirs(IND, exist_ok=True)

# Consistency: probabilities must come from the same dataset and tuned run
S1 = json.load(open(os.path.join(OUT, "stage1_params.json")))
S4 = json.load(open(os.path.join(OUT, "stage4_meta.json")))
assert S1["data_sha"] == D.data_sha == S4["data_sha"], "dataset mismatch"

hold = pd.read_csv(os.path.join(OUT, "stage4_holdout.csv"))
probs = np.load(os.path.join(OUT, "stage4_holdout_probs.npz"))
ho_auc = dict(zip(hold.model, hold.ho_auc))
ho_ci = {r.model: (r.auc_ci_lo, r.auc_ci_hi) for r in hold.itertuples() if not pd.isna(r.auc_ci_lo)}

def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")

# ---------- 1. one ROC figure per model (hold-out, all soft-probability models)
made = []
for key in probs.files:
    name = key.replace("_", " ")          # model names contain no underscores
    prob = probs[key]
    fpr, tpr, _ = roc_curve(D.y_te, prob)
    auc = roc_auc_score(D.y_te, prob)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if name in ho_ci:
        lo, hi = ho_ci[name]
        lab = f"AUC = {auc:.3f} (95% CI {lo:.3f} to {hi:.3f})"
    else:
        lab = f"AUC = {auc:.3f}"
    ax.plot(fpr, tpr, lw=2.2, color="#1f4e79", label=lab)
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Chance")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Hold-out ROC: {name}", fontweight="bold", fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=.25)
    plt.tight_layout()
    fname = f"roc_{safe(name)}.png"
    plt.savefig(os.path.join(IND, fname), dpi=DPI); plt.close()
    made.append((name, round(auc, 4), fname))

# remove the combined ROC figures so only the individual ones are used
for old in ["roc_baselines.png", "roc_ensembles.png"]:
    p = os.path.join(FIG, old)
    if os.path.exists(p): os.remove(p)

pd.DataFrame(made, columns=["model", "holdout_auc", "file"]).to_csv(
    os.path.join(OUT, "stage5b_roc_index.csv"), index=False)

# ---------- 2. weight sensitivity: the nine 3-base configurations only -----
cv = pd.read_csv(os.path.join(OUT, "stage2_cv_summary.csv"))
S2 = json.load(open(os.path.join(OUT, "stage2_selection.json")))
PV = S2["primary_voting"]
w = cv[cv.model.str.match(r"^Weighted Soft Voting \[\d, \d, \d\]$")].copy()
assert len(w) == 9, f"expected nine 3-base weight configurations, found {len(w)}"
w = w.sort_values("rep5x10_acc")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(w.model.str.replace("Weighted Soft Voting ", ""), w.rep5x10_acc,
        color=["#d62728" if m == PV else "#1f77b4" for m in w.model], alpha=.85)
for y, v in enumerate(w.rep5x10_acc):
    ax.text(v + 0.001, y, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0.80, 0.88)
ax.set_xlabel("Repeated 5x10 CV accuracy (training data); axis starts at 0.80")
ax.set_title("Voting-weight sensitivity: nine configurations (red = selected)",
             fontweight="bold", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "weight_sensitivity.png"), dpi=DPI); plt.close()

# ---------- 3. temporal validation with readable labels -------------------
tp = pd.read_csv(os.path.join(OUT, "stage4_temporal.csv"))
short_model = {"CatBoost (tuned)": "CatBoost", PV: "Primary voting [5,1,1]",
               "Stacking LR-meta (no-passthrough)": "Stacking LR (no passthrough)",
               "Random Forest": "Random Forest"}
short_scheme = {"leave-one-season-out: train Kharif -> test Zaid": "Kharif to Zaid",
                "leave-one-season-out: train Zaid -> test Kharif": "Zaid to Kharif",
                "forward: train 2013-2018 -> test 2019-2023": "Forward 2013-18 to 2019-23"}
tp["m"] = tp.model.map(short_model); tp["s"] = tp.scheme.map(short_scheme)
pivot = tp.pivot(index="m", columns="s", values="test_accuracy")
pivot = pivot.reindex([short_model[k] for k in ["CatBoost (tuned)", PV,
                       "Stacking LR-meta (no-passthrough)", "Random Forest"]])
pivot = pivot[["Kharif to Zaid", "Zaid to Kharif", "Forward 2013-18 to 2019-23"]]
ax = pivot.plot.bar(figsize=(10, 6), rot=0, color=["#2ca02c", "#ff7f0e", "#9467bd"], alpha=.85)
ax.axhline(0.5, color="k", ls="--", lw=1)
ax.set_ylim(0, 1); ax.set_ylabel("Test accuracy"); ax.set_xlabel("")
ax.set_title("Temporal validation (fixed configurations)", fontweight="bold", fontsize=11)
ax.legend(title="Scheme", loc="upper left")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "temporal_validation.png"), dpi=DPI); plt.close()

# ---------- 4. SHAP: two panels with their own scales -----------------------
sh = pd.read_csv(os.path.join(OUT, "stage5_shap.csv"), index_col=0)
ens_col = [c for c in sh.columns if "KernelSHAP" in c][0]
cat_col = [c for c in sh.columns if "TreeSHAP" in c][0]
order = sh[ens_col].sort_values().index
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
a1.barh(order, sh.loc[order, ens_col], color="#ff7f0e", alpha=.9)
a1.set_title(f"{PV} (KernelSHAP)", fontweight="bold", fontsize=11)
a1.set_xlabel("Mean |SHAP value| (hold-out rows)")
a2.barh(order, sh.loc[order, cat_col], color="#1f77b4", alpha=.9)
a2.set_title("CatBoost (TreeSHAP)", fontweight="bold", fontsize=11)
a2.set_xlabel("Mean |SHAP value| (hold-out rows)")
for a in (a1, a2): a.grid(axis="x", alpha=.25)
fig.suptitle("Feature importance. Panels use separate scales; compare rankings, not magnitudes",
             fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_importance.png"), dpi=DPI); plt.close()

json.dump(dict(stage="stage5b", data_sha=D.data_sha, dpi=DPI,
               roc_files=len(made), versions_stage1=S1["versions"]),
          open(os.path.join(OUT, "stage5b_meta.json"), "w"), indent=2)
print(f"stage 5b done: {len(made)} individual ROC figures, weight sweep (9), temporal, SHAP panels")
