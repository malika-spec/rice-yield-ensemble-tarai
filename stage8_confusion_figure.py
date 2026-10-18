"""
Draws the confusion-matrix figure from the stage 8 results. No model is refitted.
Run:    python stage8_confusion_figure.py
Reads:  results/stage8_confusion.csv
Writes: results/figures/manuscript/FigureConfusion.png  (600 DPI)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results"
DPI = 600
df = pd.read_csv(os.path.join(OUT, "stage8_confusion.csv"))
short = {"CatBoost (tuned)": "CatBoost (tuned)",
         "Weighted Soft Voting [5, 1, 1]": "Soft voting [5,1,1] (principal)",
         "Stacking LR-meta (no-passthrough)": "Stacking LR, no passthrough",
         "Stacking RF-meta (no-passthrough)": "Stacking RF, no passthrough",
         "Hard Voting (3 base)": "Hard voting (3 base)"}

n = len(df)
ncol = 3; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.6 * nrow))
for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
    cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() + 2)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low", "High"]); ax.set_yticklabels(["Low", "High"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    ax.set_title(f"{short.get(r.model, r.model)}\naccuracy {r.accuracy*100+1e-9:.1f}%", fontsize=9, fontweight="bold")
for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.tight_layout()
os.makedirs(os.path.join(OUT, "figures", "manuscript"), exist_ok=True)
plt.savefig(os.path.join(OUT, "figures", "manuscript", "FigureConfusion.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("written: results/figures/manuscript/FigureConfusion.png")
