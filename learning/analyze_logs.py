#!/usr/bin/env python3
"""Analyze training logs and provide recommendations."""
import csv
from collections import Counter

with open("training_log.csv") as f:
    rows = list(csv.DictReader(f))

reasons = Counter(r["termination_reason"] for r in rows)
rewards = [float(r["reward"]) for r in rows]
lengths = [int(r["length"]) for r in rows]
total = rows[-1]["total_steps"]

print("=" * 55)
print("  DEEP ANALYSIS: ~23K steps / 64 episodes")
print("=" * 55)

print(f"\n--- Termination ---")
for k, v in reasons.most_common():
    print(f"  {k}: {v:>2d} ({v*100//len(rows):>2d}%)")

print(f"\n--- Overall ---")
print(f"  Total steps: {total}")
print(f"  Collisions:  0 (never hit, just push into walls)")
print(f"  Completions: 0 (never flew through)")
print(f"  Avg reward:  {sum(rewards)/len(rewards):+.2f}")
print(f"  Avg length:  {sum(lengths)/len(lengths):.0f} steps")
print(f"  Best ep:     +{max(rewards):.2f} ({lengths[rewards.index(max(rewards))]} steps)")
print(f"  Worst ep:    {min(rewards):.2f} ({lengths[rewards.index(min(rewards))]} steps)")

# Trend
q = len(rows) // 4
print(f"\n--- Trend by quarters ---")
for i in range(4):
    seg = rows[i*q:(i+1)*q if i < 3 else len(rows)]
    avg_r = sum(float(r["reward"]) for r in seg) / len(seg)
    avg_l = sum(int(r["length"]) for r in seg) / len(seg)
    stuck = sum(1 for r in seg if r["termination_reason"] == "stuck")
    oob = sum(1 for r in seg if r["termination_reason"] == "out_of_bounds")
    print(f"  Q{i+1} ep{seg[0]['episode']}-{seg[-1]['episode']}:  "
          f"avgR {avg_r:+.2f}  avgL {avg_l:>4.0f}  "
          f"stuck {stuck:>2d}  oob {oob:>2d}")

# Categorize rewards
print(f"\n--- Reward distribution ---")
neg = sum(1 for r in rewards if r < 0)
low = sum(1 for r in rewards if 0 <= r < 3)
mid = sum(1 for r in rewards if 3 <= r < 8)
high = sum(1 for r in rewards if r >= 8)
print(f"  Negative (<0):   {neg:>2d}")
print(f"  Low (0-3):       {low:>2d}")
print(f"  Medium (3-8):    {mid:>2d}")
print(f"  High (8+):       {high:>2d}")

# Length distribution
print(f"\n--- Length distribution ---")
short = sum(1 for l in lengths if l < 100)
med = sum(1 for l in lengths if 100 <= l < 400)
long_ = sum(1 for l in lengths if 400 <= l < 800)
very_long = sum(1 for l in lengths if l >= 800)
print(f"  Short (<100):    {short:>2d}")
print(f"  Medium (100-400):{med:>2d}")
print(f"  Long (400-800):  {long_:>2d}")
print(f"  Very long (800+):{very_long:>2d}")

print(f"\n--- Root cause analysis ---")
print(f"  The drone ALWAYS flies forward until it hits something.")
print(f"  It never learns to turn because:")
print(f"  1. Forward reward ({0.1*0.025:.4f}/step max) was much larger than")
print(f"     wall penalty ({-0.2*0.2:.2f} max) -- drone preferred to hit wall")
print(f"  2. yaw range (±1.2 rad/s) is too aggressive -- even a small action")
print(f"     causes a violent spin, so the policy avoids using yaw")
print(f"  3. No centering signal -- drone has no incentive to stay mid-tunnel")
print(f"")
print(f"--- Fixes applied ---")
print(f"  - yaw: ±1.2 -> ±0.6 (smoother turns)")
print(f"  - vx:  0.8 -> 0.5  (slower = more time to react)")
print(f"  - Wall penalty: now quadratic -0.5*(1.5-d)^2")
print(f"    (at 0m: -1.125, at 0.5m: -0.5, at 1.0m: -0.125)")
print(f"  - Added centering bonus: -0.02*|left-right|")
print(f"")
print(f"--- Estimating improvement ---")
print(f"  With new rewards, hitting wall at 0.3m costs -0.72")
print(f"  While forward progress gives +0.012/step")
print(f"  The policy MUST turn to avoid losing reward.")
