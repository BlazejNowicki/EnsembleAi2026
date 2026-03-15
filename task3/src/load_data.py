"""
Chunked loader for large data.csv — no full read into memory.
Streams in chunks, keeps every Nth row, saves to parquet.

Usage:
    python load_data.py
    
Then in your EDA / training scripts replace:
    df = pd.read_csv(DATA_PATH, ...)
with:
    df = pd.read_parquet("eda_sample.parquet")
"""

import gc
import os
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "/net/tscratch/people/tutorial237/EnsembleAi2026/task3/data/data.csv"
DEVICES_PATH = "/net/tscratch/people/tutorial237/EnsembleAi2026/task3/data/devices.csv"
CHUNK_SIZE   = 500_000   # rows per chunk — lower if RAM is tight
KEEP_EVERY_N = 500_000         # 1-in-6 → 30-min resolution, ~17% of data
                         # use 12 for 1-hour resolution if still too big
OUT_SAMPLE   = "/net/tscratch/people/tutorial237/EnsembleAi2026/task3/data/eda_sample.parquet"
# ───────────────────────────────────────────────────────────────────────────────

TEMP_COLS = [f"t{i}" for i in range(1, 14)]
DTYPES    = {c: "float32" for c in TEMP_COLS + ["x1", "x2", "x3"]}
DTYPES.update({"deviceId": "str", "deviceType": "int8"})


def build_sample():
    chunks     = []
    total_read = 0

    reader = pd.read_csv(
        DATA_PATH,
        dtype=DTYPES,
        parse_dates=["timedate"],
        chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader):
        sampled = chunk.iloc[::KEEP_EVERY_N].copy()
        chunks.append(sampled)
        total_read += len(chunk)
        kept = sum(len(c) for c in chunks)
        print(f"  chunk {i+1:>3}: {len(chunk):>8,} read | "
              f"{len(sampled):>7,} kept | "
              f"total read: {total_read:>10,} | total kept: {kept:>8,}")
        gc.collect()

    df = pd.concat(chunks, ignore_index=True)
    df.sort_values(["deviceId", "timedate"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    print(f"Streaming {DATA_PATH} (1-in-{KEEP_EVERY_N} rows) ...")
    df = build_sample()

    print("\nMerging devices metadata ...")
    devices = pd.read_csv(DEVICES_PATH)   # devices.csv is tiny, fine to read fully
    df = df.merge(devices, on="deviceId", how="left")

    df.to_parquet(OUT_SAMPLE, index=False)

    size_mb = os.path.getsize(OUT_SAMPLE) / 1024 / 1024
    print(f"\nDone.")
    print(f"  Rows        : {len(df):,}")
    print(f"  Devices     : {df['deviceId'].nunique()}")
    print(f"  Date range  : {df['timedate'].min()} → {df['timedate'].max()}")
    print(f"  Saved to    : {OUT_SAMPLE}  ({size_mb:.1f} MB)")