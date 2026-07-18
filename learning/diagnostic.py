#!/usr/bin/env python3
"""
Diagnostic tool: shows what the drone "sees" through stereo cameras.
Reads training_log.csv and curriculum_metrics.json.
"""

import csv
import json
from pathlib import Path

LEARNING_DIR = Path(__file__).parent

def analyze_stereo_distances():
    """Analyze stereo distance patterns from training log."""
    log_file = LEARNING_DIR / "training_log.csv"
    if not log_file.exists():
        print("No training_log.csv found")
        return
    
    episodes = []
    with open(log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(row)
    
    if not episodes:
        print("No episodes found")
        return
    
    print(f"Total episodes: {len(episodes)}")
    print(f"Last 10 episodes:")
    print(f"{'Ep':>4} | {'Steps':>6} | {'Reward':>8} | {'Len':>5} | {'Reason'}")
    print("-" * 50)
    for ep in episodes[-10:]:
        print(f"{ep['episode']:>4} | {ep['total_steps']:>6} | {float(ep['reward']):>+8.2f} | {ep['length']:>5} | {ep['termination_reason']}")
    
    # Analyze reward distribution
    rewards = [float(ep['reward']) for ep in episodes]
    lengths = [int(ep['length']) for ep in episodes]
    
    print(f"\nReward statistics:")
    print(f"  Mean: {sum(rewards)/len(rewards):.2f}")
    print(f"  Min: {min(rewards):.2f}")
    print(f"  Max: {max(rewards):.2f}")
    
    print(f"\nLength statistics:")
    print(f"  Mean: {sum(lengths)/len(lengths):.0f}")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    
    # Count termination reasons
    reasons = {}
    for ep in episodes:
        r = ep['termination_reason']
        reasons[r] = reasons.get(r, 0) + 1
    
    print(f"\nTermination reasons:")
    for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {count} ({count/len(episodes)*100:.1f}%)")

def analyze_curriculum():
    """Analyze curriculum metrics."""
    metrics_file = LEARNING_DIR / "curriculum_metrics.json"
    if not metrics_file.exists():
        print("No curriculum_metrics.json found")
        return
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    stage = data.get("current_stage", 0)
    total_steps = data.get("total_steps", 0)
    episodes = data.get("episode_results", [])
    
    print(f"\nCurriculum:")
    print(f"  Stage: {stage}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Episodes: {len(episodes)}")
    
    if episodes:
        rewards = [e.get("reward", 0) for e in episodes]
        lengths = [e.get("length", 0) for e in episodes]
        
        print(f"  Avg reward: {sum(rewards)/len(rewards):.2f}")
        print(f"  Avg length: {sum(lengths)/len(lengths):.0f}")
        
        # Count reasons
        reasons = {}
        for e in episodes:
            r = e.get("termination_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        
        print(f"  Termination reasons:")
        for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {r}: {count} ({count/len(episodes)*100:.1f}%)")

if __name__ == "__main__":
    print("=" * 50)
    print("DRONE TRAINING DIAGNOSTIC")
    print("=" * 50)
    analyze_stereo_distances()
    analyze_curriculum()
