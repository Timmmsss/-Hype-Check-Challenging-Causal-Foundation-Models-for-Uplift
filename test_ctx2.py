import sys
import time
sys.path.insert(0, "/home/jupyter/project/baseline_benchmark")
import pandas as pd
from baseline_benchmark.causalpfn import CausalPFNEstimator

df_feat = pd.read_parquet("/home/jupyter/project/data_A_cleaned/Retailhero-uplift/features.parquet")
df_out = pd.read_parquet("/home/jupyter/project/data_A_cleaned/Retailhero-uplift/outcomes.parquet")

n = 20000
X = df_feat.drop(columns=[c for c in ["epk_id", "T", "treatment_dt", "split"] if c in df_feat.columns]).values[:n]
t = df_feat["T"].values[:n]
y = df_out["Y"].values[:n] if "Y" in df_out.columns else df_out.iloc[:, -1].values[:n]

X_eval = df_feat.drop(columns=[c for c in ["epk_id", "T", "treatment_dt", "split"] if c in df_feat.columns]).values[n:n+10000]

for ctx in [4096, 1024]:
    est = CausalPFNEstimator(seed=0, max_context_length=ctx)
    t0 = time.time()
    est.fit(X, t, y)
    t1 = time.time()
    preds = est.predict_cate(X_eval)
    t2 = time.time()
    print(f"ctx={ctx}: fit_seconds={t1-t0:.1f} predict_seconds={t2-t1:.1f}")
    
