import glob
import pandas as pd

paths = glob.glob("/home/jupyter/project/baseline_benchmark/results/retailhero/**/*.csv", recursive=True)
paths += glob.glob("/home/jupyter/project/baseline_benchmark/**/*.csv", recursive=True)
seen = set()
for p in paths:
    if p in seen:
        continue
    seen.add(p)
    try:
        df = pd.read_csv(p, nrows=5)
    except Exception:
        continue
    cols = [c for c in df.columns if "fit_second" in c.lower() or "model" in c.lower() or "predict_second" in c.lower()]
    if cols and any("fit_second" in c.lower() for c in df.columns):
        print(p)
        try:
            full = pd.read_csv(p)
            mask = full["model"].astype(str).str.contains("causalpfn", case=False, na=False) if "model" in full.columns else None
            if mask is not None and mask.any():
                print(full.loc[mask, [c for c in ["model","fit_seconds","predict_seconds","n_train"] if c in full.columns]])
        except Exception as e:
            print("  error:", e)
        print("---")
