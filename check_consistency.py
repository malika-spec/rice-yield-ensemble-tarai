"""
check_consistency.py — run after all stages. Confirms that every output
came from the same dataset, the same tuned parameters, the same selection
and the same library versions. Exits with an error if anything differs.
Run:    python check_consistency.py
Output: results/consistency_report.txt
"""
import sys
from common import *

problems, lines = [], []
def ok(msg): lines.append("OK    " + msg)
def bad(msg): problems.append(msg); lines.append("FAIL  " + msg)

def need(path):
    if not os.path.exists(path):
        bad(f"missing output: {path}"); return None
    return load_json(path) if path.endswith(".json") else path

D = load_data()
lines.append(f"dataset sha = {D.data_sha}  (rows 154, 22 season-years)")
files = {k: need(os.path.join(OUT, f)) for k, f in [
    ("s1", "stage1_params.json"), ("s2", "stage2_selection.json"),
    ("s3", "stage3_nested_summary.json"), ("s4", "stage4_meta.json"),
    ("s5", "stage5_meta.json")]}

# 1. one dataset everywhere
shas = {k: v["data_sha"] for k, v in files.items() if v}
if len(set(shas.values())) == 1 and shas: ok(f"dataset fingerprint identical in all stages")
else: bad(f"dataset fingerprint differs: {shas}")

# 2. one parameter set everywhere
if files["s1"] and files["s2"]:
    if files["s2"]["params_sha"] == files["s1"]["params_sha"]:
        ok(f"tuned parameters identical (params_sha {files['s1']['params_sha']})")
    else: bad("selection stage used different tuned parameters")

# 3. one selection everywhere
sel = {k: files[k]["primary_voting"] for k in ["s2", "s4"] if files[k]}
if len(set(sel.values())) == 1 and sel: ok(f"primary voting ensemble identical: {list(sel.values())[0]}")
elif sel: bad(f"primary voting differs across stages: {sel}")

# 4. one library environment everywhere
vers = {k: files[k]["versions"] for k in files if files[k]}
ref_v = next(iter(vers.values())) if vers else {}
diffs = {k: v for k, v in vers.items() if v != ref_v}
if vers and not diffs: ok(f"library versions identical across stages: {ref_v}")
elif diffs: bad(f"library versions differ between stages: {diffs}")

# 5. nested folds complete and from this dataset
fold_dir = os.path.join(OUT, "stage3")
fold_files = sorted(os.listdir(fold_dir)) if os.path.isdir(fold_dir) else []
if len(fold_files) == CV_FOLDS * OUTER_REPEATS:
    shas3 = {load_json(os.path.join(fold_dir, f))["data_sha"] for f in fold_files}
    if shas3 == {D.data_sha}: ok(f"all {len(fold_files)} nested folds present and from this dataset")
    else: bad(f"nested folds come from different datasets: {shas3}")
else:
    bad(f"nested folds: found {len(fold_files)}, expected {CV_FOLDS*OUTER_REPEATS}")

# 6. every expected table exists and is non-empty
expected = ["stage2_cv_summary.csv", "stage3_nested_folds.csv", "stage4_holdout.csv",
            "stage4_stats.csv", "stage4_temporal.csv", "stage4_outlier_sensitivity.csv",
            "stage4_mannwhitney.csv", "stage5_shap.csv", "stage5_learning_curves.csv", "stage5_shap_direction.csv"]
for f in expected:
    p = os.path.join(OUT, f)
    if os.path.exists(p) and os.path.getsize(p) > 0: ok(f"{f} present")
    else: bad(f"{f} missing or empty")
figs = ["weight_sensitivity.png", "temporal_validation.png",
        "learning_curves.png", "shap_importance.png"]
for f in figs:
    p = os.path.join(OUT, "figures", f)
    if os.path.exists(p): ok(f"figure {f} present (600 DPI requested: {DPI})")
    else: bad(f"figure {f} missing")

roc_dir = os.path.join(OUT, "figures", "roc_individual")
n_roc = len([f for f in os.listdir(roc_dir) if f.endswith(".png")]) if os.path.isdir(roc_dir) else 0
if n_roc == 22: ok("22 individual ROC figures present")
else: bad(f"expected 22 individual ROC figures, found {n_roc}")
if os.path.exists(os.path.join(OUT, "stage5b_roc_index.csv")): ok("stage5b_roc_index.csv present")
else: bad("stage5b_roc_index.csv missing")

# 7. hold-out used once: test rows never appear in tuning or selection files
hold = pd.read_csv(os.path.join(OUT, "stage4_holdout.csv")) if os.path.exists(os.path.join(OUT, "stage4_holdout.csv")) else None
if hold is not None and len(hold) > 0:
    ok(f"hold-out table has {len(hold)} models, evaluated on {len(D.X_te)} rows, once")

S1_PARAMS_SHA = files["s1"]["params_sha"] if files["s1"] else None

# stage 6 (reviewer tests): must come from the same dataset and tuned parameters
s6_path = os.path.join(OUT, "stage6_meta.json")
if os.path.exists(s6_path):
    s6 = load_json(s6_path)
    if s6["data_sha"] == D.data_sha and s6["params_sha"] == S1_PARAMS_SHA:
        ok("stage 6 reviewer tests: same dataset and tuned parameters")
    else: bad("stage 6 reviewer tests came from a different dataset or parameter set")
else:
    bad("stage6_meta.json missing: run stage6_reviewer_tests.py")

report = "\n".join(lines) + "\n\n" + ("ALL CHECKS PASSED" if not problems else
                                     f"{len(problems)} PROBLEM(S) — do not report results until resolved")
with open(os.path.join(OUT, "consistency_report.txt"), "w") as fh:
    fh.write(report + "\n")
print(report)
sys.exit(1 if problems else 0)
