"""Does any autoencoder architecture close the gap to raw features?

E9 searched the classifiers only, so the report's central comparison still rests
on one AE configuration (64 latent dimensions, 3 conv blocks). This sweeps the
architecture instead. It is a sweep, not a nested selection, and is reported as
such: the question is not which setting wins but whether ANY setting reaches the
raw-feature baseline. If none does, the conclusion holds regardless of how a
selection protocol would have chosen, which is a stronger statement than a
tuned single number.
"""
import os, subprocess, itertools, pandas as pd
GRID = [(16, 3), (32, 3), (64, 2), (64, 3), (64, 4), (128, 3)]
rows = []
for dim, layers in GRID:
    out = f"./artifacts_arch_d{dim}_l{layers}"
    env = dict(os.environ, NETMOB_ARTIFACT_DIR=out,
               NETMOB_LATENT_DIM=str(dim), NETMOB_CNN_LAYERS=str(layers))
    print(f"\n=== latent={dim} layers={layers} ===", flush=True)
    subprocess.run([".venv/bin/python", "-u", "run_notebook.py",
                    "ML_autoencoder_cnn.ipynb"], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        d = pd.read_csv(f"{out}/results.csv")
        d = d[(d.fold != "mean") & (d.model == "AE+XGBoost")]
        rows.append({"latent": dim, "layers": layers,
                     "roc_auc": d.roc_auc.astype(float).mean(),
                     "pr_auc": d.pr_auc.astype(float).mean()})
        print(f"  AE+XGBoost ROC-AUC {rows[-1]['roc_auc']:.4f}  PR-AUC {rows[-1]['pr_auc']:.4f}")
    except Exception as e:
        print("  FAILED:", e)
r = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
os.makedirs("./artifacts_arch", exist_ok=True)
r.to_csv("./artifacts_arch/sweep.csv", index=False)
print("\n=== architecture sweep, AE+XGBoost ===")
print(r.round(4).to_string(index=False))
print("\nraw-feature reference: RAW-STATS+XGBoost 0.8459 ROC-AUC / 0.6370 PR-AUC")
print(f"best AE config: {r.roc_auc.max():.4f} ROC-AUC -> "
      f"{'CLOSES' if r.roc_auc.max() >= 0.8459 else 'does NOT close'} the gap")
