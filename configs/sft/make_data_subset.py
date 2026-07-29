#!/usr/bin/env python3
"""Build a domain-balanced subset of OpenThoughts3-1.2M by selecting whole shards.

OT3's 120 parquet shards are ordered by domain (shards 0-24 code, 25-109 math,
110-119 science), so taking the first N rows would drop science entirely. Taking
every Nth shard preserves the original 850k/250k/100k mix.

python configs/sft/make_data_subset.py \
    --src /mnt/hdfs/wdl/data/OpenThoughts3-1.2M/data \
    --dst /mnt/hdfs/wdl/data/OpenThoughts3-1.2M/data_subset40 \
    --step 3 --mode copy
"""

import argparse
import os
import re
import shutil
from collections import Counter

# Shard ranges per domain, derived from the published 250k/850k/100k composition
# and verified against shards 0, 60 and 110.
DOMAIN_RANGES = [("code", 0, 24), ("math", 25, 109), ("science", 110, 119)]


def domain_of(shard_idx):
    for name, lo, hi in DOMAIN_RANGES:
        if lo <= shard_idx <= hi:
            return name
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Directory holding the 120 train-*.parquet shards.")
    parser.add_argument("--dst", default=None, help="Output directory. Omit with --dry-run.")
    parser.add_argument("--step", type=int, default=3, help="Keep every Nth shard (default 3 -> 40 shards).")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    shards = sorted(f for f in os.listdir(args.src) if f.endswith(".parquet"))
    if not shards:
        raise SystemExit(f"no .parquet files under {args.src}")

    indexed = []
    for name in shards:
        m = re.search(r"train-(\d+)-of-(\d+)", name)
        if not m:
            raise SystemExit(f"unexpected shard name: {name}")
        indexed.append((int(m.group(1)), name))
    indexed.sort()

    picked = [(i, n) for i, n in indexed if i % args.step == 0]

    mix = Counter(domain_of(i) for i, _ in picked)
    total_rows = len(picked) * 10_000
    print(f"found {len(indexed)} shards, keeping every {args.step} -> {len(picked)} shards")
    print(f"approx rows: {total_rows:,}")
    for name, _, _ in DOMAIN_RANGES:
        n = mix.get(name, 0)
        pct = 100 * n / len(picked) if picked else 0
        print(f"  {name:<8} {n:>3} shards  ~{n * 10_000:>7,} rows  ({pct:4.1f}%)")
    if mix.get("unknown"):
        print(f"  unknown  {mix['unknown']} shards (shard count differs from 120?)")

    if args.dry_run:
        return
    if not args.dst:
        raise SystemExit("--dst is required unless --dry-run")

    os.makedirs(args.dst, exist_ok=True)
    for _, name in picked:
        src = os.path.abspath(os.path.join(args.src, name))
        dst = os.path.join(args.dst, name)
        if os.path.lexists(dst):
            continue
        if args.mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"\n{args.mode}ed {len(picked)} shards into {args.dst}")
    print("point dataset_info.json's file_name at that directory")


if __name__ == "__main__":
    main()
