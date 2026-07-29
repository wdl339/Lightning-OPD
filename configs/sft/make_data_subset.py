#!/usr/bin/env python3
"""Build a subset of OpenThoughts3-1.2M by selecting whole parquet shards.

OT3's 120 shards are ordered by domain (0-24 code, 25-109 math, 110-119 science),
and sources appear to be blocked within a domain too, so any selection should be
spread evenly across the range of interest rather than taken contiguously.

    # all three domains, keeping the 850k/250k/100k mix
    python make_data_subset.py --src <dir> --step 3 --dry-run

    # math only, enough for 300 steps
    python make_data_subset.py --src <dir> --domain math --num-shards 5 --dry-run

    # materialise
    python configs/sft/make_data_subset.py --src /mnt/hdfs/wdl/data/OpenThoughts3-1.2M/data \
        --dst /mnt/hdfs/wdl/data/OpenThoughts3-1.2M/data_math50k \
        --domain math --num-shards 5 --mode copy
"""

import argparse
import os
import re
import shutil
from collections import Counter

# Verified against shards 0 (code), 60 (math) and 110 (science).
DOMAIN_RANGES = {"code": (0, 24), "math": (25, 109), "science": (110, 119)}

ROWS_PER_SHARD = 10_000
# Measured with the Qwen3 tokenizer at cutoff_len=16384: ~15.4k usable tokens per
# row, and packing fits roughly one row per bucket at this length.
EFFECTIVE_TOKENS_PER_ROW = 15_400


def domain_of(idx):
    for name, (lo, hi) in DOMAIN_RANGES.items():
        if lo <= idx <= hi:
            return name
    return "unknown"


def spread(lo, hi, count):
    """Pick `count` evenly spaced integers across the inclusive range [lo, hi]."""
    span = hi - lo
    if count <= 1:
        return [lo]
    return sorted({lo + round(span * i / (count - 1)) for i in range(count)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Directory holding the train-*.parquet shards.")
    parser.add_argument("--dst", default=None)
    parser.add_argument("--domain", choices=[*DOMAIN_RANGES, "all"], default="all")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="How many shards to keep, spread evenly across the domain range.")
    parser.add_argument("--step", type=int, default=None,
                        help="Alternative to --num-shards: keep every Nth shard.")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="copy")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if (args.num_shards is None) == (args.step is None):
        raise SystemExit("pass exactly one of --num-shards or --step")

    available = {}
    for name in os.listdir(args.src):
        m = re.search(r"train-(\d+)-of-(\d+)", name)
        if m and name.endswith(".parquet"):
            available[int(m.group(1))] = name
    if not available:
        raise SystemExit(f"no train-*.parquet under {args.src}")

    if args.domain == "all":
        lo, hi = min(available), max(available)
    else:
        lo, hi = DOMAIN_RANGES[args.domain]

    if args.num_shards is not None:
        wanted = spread(lo, hi, args.num_shards)
    else:
        wanted = [i for i in range(lo, hi + 1) if i % args.step == 0]

    picked = [(i, available[i]) for i in wanted if i in available]
    missing = [i for i in wanted if i not in available]

    rows = len(picked) * ROWS_PER_SHARD
    tokens = rows * EFFECTIVE_TOKENS_PER_ROW
    print(f"domain={args.domain}  shard range {lo}-{hi}  picked {len(picked)} shards")
    print(f"  shards: {[i for i, _ in picked]}")
    if missing:
        print(f"  MISSING (not downloaded yet): {missing}")
    print(f"  approx rows   : {rows:,}")
    print(f"  approx tokens : {tokens / 1e6:.0f}M")
    mix = Counter(domain_of(i) for i, _ in picked)
    if len(mix) > 1:
        for name in DOMAIN_RANGES:
            if mix.get(name):
                print(f"    {name:<8} {mix[name]:>3} shards  ~{mix[name] * ROWS_PER_SHARD:>7,} rows")

    for gbs_tokens, label in [(2 * 2 * 32 * 16384, "2x2x32 GPUs @ 16384")]:
        print(f"  covers {tokens / gbs_tokens:.0f} steps at {label}")

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
        os.symlink(src, dst) if args.mode == "symlink" else shutil.copy2(src, dst)
    print(f"\n{args.mode}ed {len(picked)} shards into {args.dst}")


if __name__ == "__main__":
    main()
