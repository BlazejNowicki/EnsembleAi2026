"""
Memory-efficient per-device 5-minute Ridge regression.

Instead of storing raw data per device, accumulates sufficient statistics
(XtX and Xty matrices) which are tiny regardless of data size.
Ridge solution: (XtX + alpha*I)^-1 Xty

Two passes through CSV:
  Pass 1: accumulate XtX, Xty per device from training rows
  Pass 2: predict x2 for forecast rows, average to monthly means
"""

import gc
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_FOLDER = '/net/tscratch/people/tutorial237/EnsembleAi2026/task3/data'
DATA_PATH    = f"{DATA_FOLDER}/data.csv"
DEVICES_PATH = f"{DATA_FOLDER}/devices.csv"
CHUNK_SIZE   = 500_000
OUT_PATH     = "data/submission_perdevice_5min.csv"
ALPHA        = 1.0    # Ridge regularisation
# ───────────────────────────────────────────────────────────────────────────────

FEATURES_5MIN = ["t1", "t4", "t5", "t8", "t11", "t2", "t9", "t12", "t13"]
N_FEAT        = len(FEATURES_5MIN) + 1   # +1 for intercept

DTYPES = {c: "float32" for c in [f"t{i}" for i in range(1,14)] + ["x1","x2","x3"]}
DTYPES.update({"deviceId": "str", "deviceType": "int8"})


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Append column of ones for intercept term."""
    return np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])


def solve_ridge(XtX: np.ndarray, Xty: np.ndarray, alpha: float) -> np.ndarray:
    """Solve Ridge: (XtX + alpha*I)^-1 Xty"""
    A = XtX + alpha * np.eye(N_FEAT, dtype=np.float64)
    try:
        return np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, Xty, rcond=None)[0]


# ── Pass 1: accumulate sufficient statistics ───────────────────────────────────
print("Pass 1: accumulating sufficient statistics per device...")
t_start = time.time()

# XtX: device → N_FEAT × N_FEAT  (float64 for numerical stability)
# Xty: device → N_FEAT
# n  : device → int (row count)
device_XtX  = defaultdict(lambda: np.zeros((N_FEAT, N_FEAT), dtype=np.float64))
device_Xty  = defaultdict(lambda: np.zeros(N_FEAT,           dtype=np.float64))
device_n    = defaultdict(int)

# Also track per-device feature mean/std for normalisation
device_sum  = defaultdict(lambda: np.zeros(N_FEAT-1, dtype=np.float64))
device_sum2 = defaultdict(lambda: np.zeros(N_FEAT-1, dtype=np.float64))

for i, chunk in enumerate(
    pd.read_csv(DATA_PATH, dtype=DTYPES, chunksize=CHUNK_SIZE)
):
    chunk["year"]  = chunk["timedate"].str[:4].astype("int16")
    chunk["month"] = chunk["timedate"].str[5:7].astype("int8")

    # Training rows: Oct 2024 - Apr 2025, x2 known and not null
    train_mask = (
        chunk["x2"].notna() &
        (
            ((chunk["year"] == 2024) & (chunk["month"] >= 10)) |
            ((chunk["year"] == 2025) & (chunk["month"] <= 4))
        )
    )
    train_chunk = chunk[train_mask].dropna(subset=FEATURES_5MIN)

    if len(train_chunk) == 0:
        del chunk; gc.collect(); continue

    for did, grp in train_chunk.groupby("deviceId"):
        X = grp[FEATURES_5MIN].values.astype(np.float64)
        y = grp["x2"].values.astype(np.float64)
        Xb = add_intercept(X)

        device_XtX[did]  += Xb.T @ Xb
        device_Xty[did]  += Xb.T @ y
        device_n[did]    += len(y)
        device_sum[did]  += X.sum(axis=0)
        device_sum2[did] += (X**2).sum(axis=0)

    print(f"  chunk {i+1:>3}: {train_mask.sum():>7,} train rows | "
          f"elapsed: {(time.time()-t_start)/60:.1f}min")
    del chunk, train_chunk; gc.collect()

print(f"\nDevices with data: {len(device_XtX)}")
print(f"Pass 1 done in {(time.time()-t_start)/60:.1f} min")


# ── Fit Ridge models (solve linear system per device) ─────────────────────────
print("\nSolving Ridge per device...")
device_weights = {}

for did in device_XtX:
    if device_n[did] < 10:
        continue
    w = solve_ridge(device_XtX[did], device_Xty[did], ALPHA)
    device_weights[did] = w

print(f"Models solved: {len(device_weights)}")

# Free sufficient statistics
del device_XtX, device_Xty, device_sum, device_sum2
gc.collect()


# ── Pass 2: predict forecast rows ─────────────────────────────────────────────
print("\nPass 2: predicting forecast rows...")

device_month_sum   = defaultdict(lambda: defaultdict(float))
device_month_count = defaultdict(lambda: defaultdict(int))

# Global fallback: accumulate global monthly means from training
global_month_sum   = defaultdict(float)
global_month_count = defaultdict(int)

for i, chunk in enumerate(
    pd.read_csv(DATA_PATH, dtype=DTYPES, chunksize=CHUNK_SIZE)
):
    chunk["year"]  = chunk["timedate"].str[:4].astype("int16")
    chunk["month"] = chunk["timedate"].str[5:7].astype("int8")

    # Forecast rows: May-Oct 2025
    fore_mask = (
        chunk["x2"].isna() &
        (chunk["year"] == 2025) &
        (chunk["month"].between(5, 10))
    )
    fore_chunk = chunk[fore_mask].dropna(subset=FEATURES_5MIN)

    if len(fore_chunk) > 0:
        for did, grp in fore_chunk.groupby("deviceId"):
            if did not in device_weights:
                continue
            X  = grp[FEATURES_5MIN].values.astype(np.float64)
            Xb = add_intercept(X)
            w  = device_weights[did]
            p  = (Xb @ w).clip(0, 1)

            for month, month_grp in grp.groupby("month"):
                idx = grp.index.get_indexer(month_grp.index)
                device_month_sum[did][month]   += p[idx].sum()
                device_month_count[did][month] += len(idx)

    # Also collect training monthly means for global fallback
    train_mask = (
        chunk["x2"].notna() &
        (
            ((chunk["year"] == 2024) & (chunk["month"] >= 10)) |
            ((chunk["year"] == 2025) & (chunk["month"] <= 4))
        )
    )
    if train_mask.sum() > 0:
        tr = chunk[train_mask]
        for month, grp in tr.groupby("month"):
            global_month_sum[month]   += grp["x2"].sum()
            global_month_count[month] += len(grp)

    print(f"  chunk {i+1:>3}: {fore_mask.sum():>7,} fore rows | "
          f"elapsed: {(time.time()-t_start)/60:.1f}min")
    del chunk; gc.collect()


# ── Aggregate to monthly means ─────────────────────────────────────────────────
print("\nAggregating to monthly means...")
rows = []
for did, month_dict in device_month_sum.items():
    for month, s in month_dict.items():
        n = device_month_count[did][month]
        if n > 0:
            rows.append({
                "deviceId"  : did,
                "year"      : 2025,
                "month"     : month,
                "prediction": s / n
            })

submission = pd.DataFrame(rows)

# ── Handle missing devices ─────────────────────────────────────────────────────
monthly_agg = pd.read_parquet(f"{DATA_FOLDER}/monthly_agg.parquet")
fore_agg    = monthly_agg[monthly_agg["x2_mean"].isna()][["deviceId","year","month"]]

if len(submission) > 0:
    submitted = set(zip(submission["deviceId"],
                        submission["year"],
                        submission["month"]))
    missing = fore_agg[~fore_agg.apply(
        lambda r: (r["deviceId"], r["year"], r["month"]) in submitted, axis=1)]
else:
    missing = fore_agg

print(f"Missing device-months: {len(missing)}")

if len(missing) > 0:
    global_monthly = {m: global_month_sum[m] / global_month_count[m]
                      for m in global_month_sum if global_month_count[m] > 0}
    missing = missing.copy()
    missing["prediction"] = missing["month"].map(global_monthly).fillna(0.1)
    submission = pd.concat([submission, missing], ignore_index=True)

submission = (submission
              .sort_values(["deviceId","year","month"])
              .reset_index(drop=True))
submission["prediction"] = submission["prediction"].clip(0, 1)

# Sanity checks
assert submission["prediction"].between(0,1).all(), "Out of range!"
assert submission.duplicated(["deviceId","year","month"]).sum() == 0, "Duplicates!"

print(f"\nSubmission rows : {len(submission)}")
print(f"Prediction range: {submission['prediction'].min():.4f} – "
      f"{submission['prediction'].max():.4f}")
print(f"Prediction mean : {submission['prediction'].mean():.4f}")
print("\nMean prediction per month:")
print(submission.groupby(["year","month"])["prediction"].mean().round(4).to_string())

submission.to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")
print(f"Total time: {(time.time()-t_start)/60:.1f} min")