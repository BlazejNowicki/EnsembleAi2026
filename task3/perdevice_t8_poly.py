"""
Per-device 5-minute polynomial regression using t8 as primary feature.

Key insight:
- t8 (load HEX temperature) correlates 0.86-0.93 with x2
- t8 forecast values (0.492-0.505) are WITHIN training range (0.481-0.546)
- No extrapolation needed — pure interpolation

Model per device:
  x2 ~ t8 + t8² + t4 + t5 + t4² + t5²

Uses sufficient statistics (XtX, Xty) — no RAM issues.
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
OUT_PATH     = "data/submission_t8_poly.csv"
ALPHA        = 0.1    # lower regularisation — t8 relationship is strong
# ───────────────────────────────────────────────────────────────────────────────

DTYPES = {c: "float32" for c in [f"t{i}" for i in range(1,14)] + ["x1","x2","x3"]}
DTYPES.update({"deviceId": "str", "deviceType": "int8"})


def make_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build polynomial feature matrix from raw 5-min readings.
    Features: [t8, t8², t4, t4², t5, t5², t8*t4, t8*t5, 1]
    """
    t8 = df["t8"].values.astype(np.float64)
    t4 = df["t4"].values.astype(np.float64)
    t5 = df["t5"].values.astype(np.float64)
    t1 = df["t1"].values.astype(np.float64)

    X = np.column_stack([
        t8, t8**2,          # t8 primary signal
        t4, t4**2,          # load HEX temp 1
        t5, t5**2,          # load HEX temp 2
        t8 * t4,            # interaction
        t8 * t5,            # interaction
        t1,                 # outdoor temp (available in forecast)
        np.ones(len(df))    # intercept
    ])
    return X


N_FEAT = 10   # must match number of columns in make_features


def solve_ridge(XtX: np.ndarray, Xty: np.ndarray, alpha: float) -> np.ndarray:
    A = XtX + alpha * np.eye(N_FEAT, dtype=np.float64)
    try:
        return np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, Xty, rcond=None)[0]


# ── Pass 1: accumulate sufficient statistics ───────────────────────────────────
print("Pass 1: accumulating sufficient statistics...")
t_start = time.time()

device_XtX = defaultdict(lambda: np.zeros((N_FEAT, N_FEAT), dtype=np.float64))
device_Xty = defaultdict(lambda: np.zeros(N_FEAT, dtype=np.float64))
device_n   = defaultdict(int)

for i, chunk in enumerate(
    pd.read_csv(DATA_PATH, dtype=DTYPES, chunksize=CHUNK_SIZE)
):
    chunk["year"]  = chunk["timedate"].str[:4].astype("int16")
    chunk["month"] = chunk["timedate"].str[5:7].astype("int8")

    train_mask = (
        chunk["x2"].notna() &
        (
            ((chunk["year"] == 2024) & (chunk["month"] >= 10)) |
            ((chunk["year"] == 2025) & (chunk["month"] <= 4))
        )
    )
    train_chunk = chunk[train_mask].dropna(subset=["t8","t4","t5","t1","x2"])

    if len(train_chunk) == 0:
        del chunk; gc.collect(); continue

    for did, grp in train_chunk.groupby("deviceId"):
        X  = make_features(grp)
        y  = grp["x2"].values.astype(np.float64)
        device_XtX[did] += X.T @ X
        device_Xty[did] += X.T @ y
        device_n[did]   += len(y)

    print(f"  chunk {i+1:>3}: {train_mask.sum():>7,} train rows | "
          f"elapsed: {(time.time()-t_start)/60:.1f}min")
    del chunk, train_chunk; gc.collect()

print(f"\nDevices with data: {len(device_XtX)}")

# Solve Ridge per device
print("Solving per-device Ridge...")
device_weights = {}
for did in device_XtX:
    if device_n[did] < 20:
        continue
    device_weights[did] = solve_ridge(device_XtX[did], device_Xty[did], ALPHA)

print(f"Models solved: {len(device_weights)}")
del device_XtX, device_Xty; gc.collect()


# ── Pass 2: predict forecast rows ─────────────────────────────────────────────
print("\nPass 2: predicting forecast rows...")

device_month_sum   = defaultdict(lambda: defaultdict(float))
device_month_count = defaultdict(lambda: defaultdict(int))
global_month_sum   = defaultdict(float)
global_month_count = defaultdict(int)

for i, chunk in enumerate(
    pd.read_csv(DATA_PATH, dtype=DTYPES, chunksize=CHUNK_SIZE)
):
    chunk["year"]  = chunk["timedate"].str[:4].astype("int16")
    chunk["month"] = chunk["timedate"].str[5:7].astype("int8")

    # Forecast: May-Oct 2025
    fore_mask = (
        chunk["x2"].isna() &
        (chunk["year"] == 2025) &
        (chunk["month"].between(5, 10))
    )
    fore_chunk = chunk[fore_mask].dropna(subset=["t8","t4","t5","t1"])

    if len(fore_chunk) > 0:
        for did, grp in fore_chunk.groupby("deviceId"):
            if did not in device_weights:
                continue
            X = make_features(grp)
            p = (X @ device_weights[did]).clip(0, 1)
            for month, month_grp in grp.groupby("month"):
                idx = grp.index.get_indexer(month_grp.index)
                device_month_sum[did][month]   += p[idx].sum()
                device_month_count[did][month] += len(idx)

    # Collect training global monthly means for fallback
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
print("\nAggregating...")
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

# Handle missing devices
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

assert submission["prediction"].between(0,1).all()
assert submission.duplicated(["deviceId","year","month"]).sum() == 0

print(f"\nSubmission rows : {len(submission)}")
print(f"Prediction range: {submission['prediction'].min():.4f} – "
      f"{submission['prediction'].max():.4f}")
print(f"Prediction mean : {submission['prediction'].mean():.4f}")
print("\nMean prediction per month:")
print(submission.groupby(["year","month"])["prediction"].mean().round(4).to_string())

submission.to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")
print(f"Total time: {(time.time()-t_start)/60:.1f} min")