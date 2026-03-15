"""
Batched per-device neural network — all 600 devices trained simultaneously.

Instead of 600 sequential models, uses a single batched model where:
- Each sample is (device_id, features, target)
- Device embedding learned alongside the MLP
- Full GPU utilisation on A100

Architecture:
  [features(7) + device_embedding(16)] → 64 → 32 → 1 (sigmoid)
"""

import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_FOLDER = '/net/tscratch/people/tutorial237/EnsembleAi2026/task3/data'
DATA_PATH    = f"{DATA_FOLDER}/data.csv"
DEVICES_PATH = f"{DATA_FOLDER}/devices.csv"
OUT_PATH     = "data/submission_perdevice_nn.csv"
CHUNK_SIZE = 500_000
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_EPOCHS   = 300
LR         = 5e-3
BATCH_SIZE = 65536    # large batch — A100 can handle it
EMBED_DIM  = 16       # device embedding dimension
# ───────────────────────────────────────────────────────────────────────────────

print(f"Using: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DTYPES = {c: "float32" for c in [f"t{i}" for i in range(1,14)] + ["x1","x2","x3"]}
DTYPES.update({"deviceId": "str", "deviceType": "int8"})

N_FEATURES = 7


def make_features_np(df: pd.DataFrame) -> np.ndarray:
    t8 = df["t8"].values.astype(np.float32)
    t4 = df["t4"].values.astype(np.float32)
    t5 = df["t5"].values.astype(np.float32)
    t1 = df["t1"].values.astype(np.float32)
    return np.column_stack([
        t8, t8**2,
        t4, t5,
        t1,
        t8 * t4,
        t8 * t5,
    ])


class BatchedDeviceNet(nn.Module):
    """
    Single model for all devices.
    Each device gets a learned embedding concatenated to input features.
    """
    def __init__(self, n_devices: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(n_devices, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES + embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Init embeddings small
        nn.init.normal_(self.embedding.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, device_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(device_ids)
        return self.net(torch.cat([x, emb], dim=1)).squeeze(-1)


# ── Pass 1: collect training data ─────────────────────────────────────────────
print("\nPass 1: collecting training data...")
t_start = time.time()

all_X = []
all_y = []
all_d = []   # device index

device_X_list = defaultdict(list)
device_y_list = defaultdict(list)

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

    if len(train_chunk) > 0:
        for did, grp in train_chunk.groupby("deviceId"):
            device_X_list[did].append(make_features_np(grp))
            device_y_list[did].append(grp["x2"].values.astype(np.float32))

    print(f"  chunk {i+1:>3}: {train_mask.sum():>7,} rows | "
          f"elapsed: {(time.time()-t_start)/60:.1f}min")
    del chunk, train_chunk; gc.collect()

# Build device index and concatenate
devices_list = sorted(device_X_list.keys())
n_devices    = len(devices_list)
device_idx   = {did: i for i, did in enumerate(devices_list)}

for did in devices_list:
    X = np.vstack(device_X_list[did])
    y = np.concatenate(device_y_list[did])
    d = np.full(len(y), device_idx[did], dtype=np.int64)
    all_X.append(X)
    all_y.append(y)
    all_d.append(d)

all_X = np.vstack(all_X)
all_y = np.concatenate(all_y)
all_d = np.concatenate(all_d)

del device_X_list, device_y_list; gc.collect()

print(f"\nTotal training samples: {len(all_X):,}")
print(f"Devices: {n_devices}")
print(f"Pass 1 done in {(time.time()-t_start)/60:.1f} min")


# ── Normalise features ────────────────────────────────────────────────────────
feat_mean = all_X.mean(axis=0)
feat_std  = all_X.std(axis=0) + 1e-8

all_X_norm = ((all_X - feat_mean) / feat_std).astype(np.float32)
del all_X; gc.collect()


# ── Train batched model on GPU ────────────────────────────────────────────────
print(f"\nTraining batched model on {DEVICE}...")
t_train = time.time()

X_gpu = torch.tensor(all_X_norm, dtype=torch.float32).to(DEVICE)
y_gpu = torch.tensor(all_y,      dtype=torch.float32).to(DEVICE)
d_gpu = torch.tensor(all_d,      dtype=torch.long   ).to(DEVICE)

del all_X_norm, all_y, all_d; gc.collect()
torch.cuda.empty_cache()

model     = BatchedDeviceNet(n_devices).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)
criterion = nn.L1Loss()

n_total = len(X_gpu)
model.train()

for epoch in range(N_EPOCHS):
    perm       = torch.randperm(n_total, device=DEVICE)
    epoch_loss = 0.0
    n_batches  = 0

    for start in range(0, n_total, BATCH_SIZE):
        idx       = perm[start:start+BATCH_SIZE]
        X_batch   = X_gpu[idx]
        y_batch   = y_gpu[idx]
        d_batch   = d_gpu[idx]

        optimizer.zero_grad()
        pred = model(X_batch, d_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:>3}/{N_EPOCHS} | "
              f"MAE={epoch_loss/n_batches:.5f} | "
              f"elapsed: {time.time()-t_train:.1f}s")

model.eval()
print(f"Training done in {time.time()-t_train:.1f}s")

# Free GPU training data
del X_gpu, y_gpu, d_gpu; gc.collect()
torch.cuda.empty_cache()


# ── Pass 2: predict forecast rows ────────────────────────────────────────────
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

    fore_mask = (
        chunk["x2"].isna() &
        (chunk["year"] == 2025) &
        (chunk["month"].between(5, 10))
    )
    fore_chunk = chunk[fore_mask].dropna(subset=["t8","t4","t5","t1"])

    if len(fore_chunk) > 0:
        for did, grp in fore_chunk.groupby("deviceId"):
            if did not in device_idx:
                continue

            X_raw  = make_features_np(grp)
            X_norm = ((X_raw - feat_mean) / feat_std).astype(np.float32)
            X_t    = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)
            d_t    = torch.full((len(X_t),), device_idx[did],
                                dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                preds = model(X_t, d_t).cpu().numpy().clip(0, 1)

            for month, month_grp in grp.groupby("month"):
                idx_loc = grp.index.get_indexer(month_grp.index)
                device_month_sum[did][month]   += preds[idx_loc].sum()
                device_month_count[did][month] += len(idx_loc)

    # Global fallback
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


# ── Aggregate ─────────────────────────────────────────────────────────────────
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
    missing        = missing.copy()
    missing["prediction"] = missing["month"].map(global_monthly).fillna(0.1)
    submission     = pd.concat([submission, missing], ignore_index=True)

submission = (submission
              .sort_values(["deviceId","year","month"])
              .reset_index(drop=True))
submission["prediction"] = submission["prediction"].clip(0, 1)

assert submission["prediction"].between(0, 1).all()
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