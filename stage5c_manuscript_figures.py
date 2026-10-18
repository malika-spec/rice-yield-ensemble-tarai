"""
STAGE 5c — manuscript figures built from the verified hold-out outputs.
Run:    python stage5c_manuscript_figures.py      (after stage 5 and 5b; no refitting)
Reads:  results/stage4_holdout_probs.npz, results/stage4_holdout.csv, dataset
Writes: results/figures/manuscript/Figure3.png   six separate ROC panels (a-f)
        results/figures/manuscript/FigureS1.png  all soft-probability models, separate panels
Both use the same probabilities as the individual ROC figures, so the curves are identical.
Model names are used for lookup exactly as stored; only the displayed title can differ.
"""
import warnings; warnings.filterwarnings("ignore")
import os, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from common import load_data, DPI, OUT, load_json

D = load_data()
S1 = load_json(os.path.join(OUT, "stage1_params.json"))
assert S1["data_sha"] == D.data_sha, "dataset mismatch"
probs = np.load(os.path.join(OUT, "stage4_holdout_probs.npz"))
hold = pd.read_csv(os.path.join(OUT, "stage4_holdout.csv")).set_index("model")
OUTDIR = os.path.join(OUT, "figures", "manuscript"); os.makedirs(OUTDIR, exist_ok=True)
y = np.asarray(D.y_te)

# Display titles. Lookup always uses the stored name; these change only the title text.
UNWEIGHTED_NAME = "Weighted Soft Voting [1, 1, 1]"   # equal weights = unweighted soft voting
DISPLAY = {UNWEIGHTED_NAME: "Unweighted Soft Voting [1, 1, 1]"}

def curve(name):
    key = name.replace(" ", "_")
    if key not in probs.files:
        raise KeyError(f"missing hold-out probabilities for: {name}")
    p = probs[key]
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    lo, hi = hold.loc[name, "auc_ci_lo"], hold.loc[name, "auc_ci_hi"]
    ref = hold.loc[name, "ho_auc"]
    if abs(round(auc, 4) - ref) > 0.0015:
        raise ValueError(f"AUC for {name} is {auc:.4f} but the verified table gives {ref:.4f}. "
                         "The probability file does not match the verified run. Figures not written.")
    return fpr, tpr, auc, lo, hi

def panel(ax, name, label):
    fpr, tpr, auc, lo, hi = curve(name)
    ax.plot(fpr, tpr, lw=1.8, color="#1f4e79", label=f"AUC {auc:.3f}\n95% CI {lo:.3f}–{hi:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    title = DISPLAY.get(name, name)
    ax.set_title(f"{label}  {title}", fontsize=8.5, loc="left", fontweight="bold")
    ax.set_xlabel("False positive rate", fontsize=8); ax.set_ylabel("True positive rate", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=.25)
    ax.legend(fontsize=7, loc="lower right", frameon=True)

# ---------- Figure 3: six main-text panels ---------------------------------
MAIN = [
    "CatBoost (tuned)",
    "Weighted Soft Voting [5, 1, 1]",
    "Stacking LR-meta (no-passthrough)",
    "Weighted Soft Voting [4, 2, 1]",
    "Weighted Soft Voting [3,2,1,1] (4 base)",
    UNWEIGHTED_NAME,
]
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2))
for ax, name, lab in zip(axes.ravel(), MAIN, "abcdef"):
    panel(ax, name, f"({lab})")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "Figure3.png"), dpi=DPI); plt.close()

# ---------- Figure S1: every soft-probability model ------------------------
# Names come from the verified hold-out table, so each lookup name matches a stored key.
ALLNAMES = [n for n in hold.index if n.replace(" ", "_") in probs.files]
ALLNAMES = sorted(ALLNAMES, key=lambda n: (not n.startswith("CatBoost"), n))
ncol = 5; nrow = math.ceil(len(ALLNAMES) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.4 * nrow))
for i, ax in enumerate(axes.ravel()):
    if i < len(ALLNAMES):
        panel(ax, ALLNAMES[i], f"S{i+1}")
    else:
        ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "FigureS1.png"), dpi=DPI); plt.close()

print(f"Figure 3 (6 panels) and Figure S1 ({len(ALLNAMES)} panels) written to results/figures/manuscript/")
