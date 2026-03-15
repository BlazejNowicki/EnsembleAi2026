"""
Per-device batched neural network v6.
Changes from v4:
  - Residual connections (ResNet-style)
  - All temperature features: t1-t13 (except dead t3) = 12 temps
  - Batch normalisation instead of dropout
  - Larger embedding: 64-dim
  - More interactions: t8*t1, t4*t5
"""

import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
SEED        = 42
DATA_FOLDER = '/net/tscratch/people/tutorial237/EnsembleAi2026/task3/data'
DATA_PATH   = f"{DATA_FOLDER}/data.csv"
DEVICES_PATH= f"{DATA_FOLDER}/devices.csv"
OUT_PATH    = f"data/submission_nn_v6_s{SEED}.csv"
CKPT_PATH   = f"{DATA_FOLDER}/checkpoint_nn_v6_s{SEED}.pt"
# ───────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE  = 500_000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_EPOCHS    = 500
LR          = 3e-3
BATCH_SIZE  = 65536
EMBED_DIM   = 64    # was 32
N_FEATURES  = 20    # see make_features_np below

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Seed: {SEED}")
print(f"Using: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DTYPES = {c: "float32" for c in [f"t{i}" for i in range(1,14)] + ["x1","x2","x3"]}
DTYPES.update({"deviceId": "str", "deviceType": "int8"})

# All temps except dead t3
TEMP_COLS  = ["t1","t2","t4","t5","t6","t7","t8","t9","t10","t11","t12","t13"]
TRAIN_DROP = TEMP_COLS + ["x2"]
FORE_DROP  = TEMP_COLS


def make_features_np(df: pd.DataFrame) -> np.ndarray:
    t1  = df["t1"].values.astype(np.float32)
    t2  = df["t2"].values.astype(np.float32)
    t4  = df["t4"].values.astype(np.float32)
    t5  = df["t5"].values.astype(np.float32)
    t6  = df["t6"].values.astype(np.float32)
    t7  = df["t7"].values.astype(np.float32)
    t8  = df["t8"].values.astype(np.float32)
    t9  = df["t9"].values.astype(np.float32)
    t10 = df["t10"].values.astype(np.float32)
    t11 = df["t11"].values.astype(np.float32)
    t12 = df["t12"].values.astype(np.float32)
    t13 = df["t13"].values.astype(np.float32)

    dtype = df["deviceType"].values.astype(np.float32) / 19.0
    hour  = df["hour"].values.astype(np.float32)
    h_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    h_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)

    return np.column_stack([
        # Primary features
        t8, t8**2,          # strongest signal
        t4, t5,             # load HEX
        t12, t13,           # receiver HEX
        # Secondary temperatures
        t1, t2,             # outdoor + indoor
        t6, t7,             # load HEX additional
        t9, t10, t11,       # CH buffer + source air
        # Interactions
        t8 * t4,
        t8 * t5,
        t8 * t1,            # new: load × outdoor
        t4 * t5,            # new: load interaction
        # Meta
        dtype,
        h_sin, h_cos,
    ])


# N_FEATURES = 20 — verify: 2+2+2+2+4+4+3+1+2 = 20 ✓


class ResidualBlock(nn.Module):
    """Residual block with batch norm."""
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class BatchedDeviceNet(nn.Module):
    def __init__(self, n_devices: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(n_devices, embed_dim)

        input_dim = N_FEATURES + embed_dim  # 20 + 64 = 84

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)

        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        nn.init.normal_(self.embedding.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, device_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(device_ids)
        h   = self.input_proj(torch.cat([x, emb], dim=1))
        h   = self.res1(h)
        h   = self.res2(h)
        return self.output(h).squeeze(-1)


# ── Pass 1: collect training data ─────────────────────────────────────────────
print("\nPass 1: collecting training data...")
t_start = time.time()

device_X_list = defaultdict(list)
device_y_list = defaultdict(list)

for i, chunk in enumerate(
    pd.read_csv(DATA_PATH, dtype=DTYPES, chunksize=CHUNK_SIZE)
):
    chunk["year"]  = chunk["timedate"].str[:4].astype("int16")
    chunk["month"] = chunk["timedate"].str[5:7].astype("int8")
    chunk["hour"]  = chunk["timedate"].str[11:13].astype(np.float32)

    train_mask = (
        chunk["x2"].notna() &
        (
            ((chunk["year"] == 2024) & (chunk["month"] >= 10)) |
            ((chunk["year"] == 2025) & (chunk["month"] <= 4))
        )
    )
    train_chunk = chunk[train_mask].dropna(subset=TRAIN_DROP)

    if len(train_chunk) > 0:
        for did, grp in train_chunk.groupby("deviceId"):
            device_X_list[did].append(make_features_np(grp))
            device_y_list[did].append(grp["x2"].values.astype(np.float32))

    print(f"  chunk {i+1:>3}: {train_mask.sum():>7,} rows | "
          f"elapsed: {(time.time()-t_start)/60:.1f}min")
    del chunk, train_chunk; gc.collect()

devices_list = sorted(device_X_list.keys())
n_devices    = len(devices_list)
device_idx   = {did: i for i, did in enumerate(devices_list)}

all_X, all_y, all_d = [], [], []
for did in devices_list:
    X = np.vstack(device_X_list[did])
    y = np.concatenate(device_y_list[did])
    d = np.full(len(y), device_idx[did], dtype=np.int64)
    all_X.append(X); all_y.append(y); all_d.append(d)

all_X = np.vstack(all_X)
all_y = np.concatenate(all_y)
all_d = np.concatenate(all_d)

del device_X_list, device_y_list; gc.collect()

print(f"\nTotal training samples: {len(all_X):,}")
print(f"Devices: {n_devices}")
print(f"Pass 1 done in {(time.time()-t_start)/60:.1f} min")

# Normalise
feat_mean  = all_X.mean(axis=0).astype(np.float32)
feat_std   = (all_X.std(axis=0) + 1e-8).astype(np.float32)
all_X_norm = ((all_X - feat_mean) / feat_std).astype(np.float32)
del all_X; gc.collect()


# ── Train on GPU ───────────────────────────────────────────────────────────────
print(f"\nTraining on {DEVICE}...")
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

# Print model size
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

model.train()
for epoch in range(N_EPOCHS):
    perm       = torch.randperm(n_total, device=DEVICE)
    epoch_loss = 0.0
    n_batches  = 0
    for start in range(0, n_total, BATCH_SIZE):
        idx = perm[start:start+BATCH_SIZE]
        optimizer.zero_grad()
        loss = criterion(model(X_gpu[idx], d_gpu[idx]), y_gpu[idx])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1
    scheduler.step()
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1:>3}/{N_EPOCHS} | "
              f"MAE={epoch_loss/n_batches:.5f} | "
              f"elapsed: {time.time()-t_train:.1f}s")

model.eval()
print(f"Training done in {time.time()-t_train:.1f}s")

del X_gpu, y_gpu, d_gpu; gc.collect()
torch.cuda.empty_cache()

# Save checkpoint
torch.save({
    "model_state"  : model.state_dict(),
    "feat_mean"    : feat_mean,
    "feat_std"     : feat_std,
    "devices_list" : devices_list,
    "device_idx"   : device_idx,
    "n_devices"    : n_devices,
    "n_features"   : N_FEATURES,
    "embed_dim"    : EMBED_DIM,
    "seed"         : SEED,
}, CKPT_PATH)
print(f"Checkpoint saved → {CKPT_PATH}")


# ── Pass 2: predict ────────────────────────────────────────────────────────────
print("\nPass 2: predicting...")

device_month_sum   = defaultdict(lambda: defaultdict(float))
device_month_count = defaultdict(lambda: defaultdict(int))
global_month_sum   = defaultdict(float)
global_month_count = defaultdict(int)

for i, chunk in enumerate(
    pd.read_csv(DATA_PATH, dtype=DTYPES, chunksize=CHUNK_SIZE)
):
    chunk["year"]  = chunk["timedate"].str[:4].astype("int16")
    chunk["month"] = chunk["timedate"].str[5:7].astype("int8")
    chunk["hour"]  = chunk["timedate"].str[11:13].astype(np.float32)

    fore_mask = (
        chunk["x2"].isna() &
        (chunk["year"] == 2025) &
        (chunk["month"].between(5, 10))
    )
    fore_chunk = chunk[fore_mask].dropna(subset=FORE_DROP)

    if len(fore_chunk) > 0:
        for did, grp in fore_chunk.groupby("deviceId"):
            if did not in device_idx:
                continue
            X_norm = ((make_features_np(grp) - feat_mean) / feat_std).astype(np.float32)
            X_t    = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)
            d_t    = torch.full((len(X_t),), device_idx[did],
                                dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                preds = model(X_t, d_t).cpu().numpy().clip(0, 1)
            for month, month_grp in grp.groupby("month"):
                idx_loc = grp.index.get_indexer(month_grp.index)
                device_month_sum[did][month]   += preds[idx_loc].sum()
                device_month_count[did][month] += len(idx_loc)

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


# ── Aggregate ──────────────────────────────────────────────────────────────────
rows = []
for did, month_dict in device_month_sum.items():
    for month, s in month_dict.items():
        n = device_month_count[did][month]
        if n > 0:
            rows.append({"deviceId": did, "year": 2025,
                         "month": month, "prediction": s / n})

submission = pd.DataFrame(rows)

monthly_agg = pd.read_parquet(f"{DATA_FOLDER}/monthly_agg.parquet")
fore_agg    = monthly_agg[monthly_agg["x2_mean"].isna()][["deviceId","year","month"]]

if len(submission) > 0:
    submitted = set(zip(submission["deviceId"], submission["year"], submission["month"]))
    missing   = fore_agg[~fore_agg.apply(
        lambda r: (r["deviceId"], r["year"], r["month"]) in submitted, axis=1)]
else:
    missing = fore_agg

if len(missing) > 0:
    global_monthly = {m: global_month_sum[m] / global_month_count[m]
                      for m in global_month_sum if global_month_count[m] > 0}
    missing = missing.copy()
    missing["prediction"] = missing["month"].map(global_monthly).fillna(0.1)
    submission = pd.concat([submission, missing], ignore_index=True)

submission = (submission.sort_values(["deviceId","year","month"])
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