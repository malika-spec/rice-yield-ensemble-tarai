"""
Redraws the SHAP importance figure (Figure 4) with the corrected EC label.
Run:    python stage5_shap_figure.py
Reads:  results/stage5_shap.csv   (final SHAP values from stage 5; not modified)
Writes: results/figures/shap_importance.png   (600 DPI)
No model is refitted. Only the figure is regenerated; no values change.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "results"
DPI = 600

# Label corrections for display only. The CSV and the data file are not changed.
LABELS = {"EC (mmos/cm^2)": "EC (mmhos/cm)"}

shap_df = pd.read_csv(os.path.join(OUT, "stage5_shap.csv"), index_col=0)
shap_df = shap_df.rename(index=LABELS)

ens_col = [c for c in shap_df.columns if "KernelSHAP" in c][0]
cat_col = [c for c in shap_df.columns if "TreeSHAP" in c][0]

order = shap_df.sort_values(ens_col, ascending=True).index
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1]})

ax1.barh(order, shap_df.loc[order, ens_col], color="#ff8c24", alpha=0.95)
ax1.set_title(ens_col, fontsize=11, fontweight="bold")
ax1.set_xlabel("Mean |SHAP value| (hold-out rows)")
ax1.grid(axis="x", alpha=0.3)

ax2.barh(order, shap_df.loc[order, cat_col], color="#3385bd", alpha=0.95)
ax2.set_title(cat_col, fontsize=11, fontweight="bold")
ax2.set_xlabel("Mean |SHAP value| (hold-out rows)")
ax2.set_yticklabels([])
ax2.grid(axis="x", alpha=0.3)

fig.suptitle("Feature importance. Panels use separate scales; compare rankings, not magnitudes", fontsize=11)
plt.tight_layout()
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
plt.savefig(os.path.join(OUT, "figures", "shap_importance.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("written: results/figures/shap_importance.png (EC label corrected; no values changed)")
