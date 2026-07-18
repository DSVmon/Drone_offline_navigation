#!/usr/bin/env python3
"""
Autonomous training monitor.
Runs periodically, checks metrics, makes adjustments, logs actions.
"""

import json
import csv
import time
from pathlib import Path
import config


def load_curriculum():
    """Load curriculum metrics."""
    path = config.LEARNING_DIR / "curriculum_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_recent_episodes(n=50):
    """Load last N episodes from training log."""
    path = config.LEARNING_DIR / "training_log.csv"
    if not path.exists():
        return []
    episodes = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(row)
    return episodes[-n:]


def analyze(episodes, curriculum):
    """Analyze training state and return recommendations."""
    if not episodes:
        return {"status": "no_data", "actions": []}

    actions = []
    issues = []

    # Calculate metrics
    rewards = [float(e["reward"]) for e in episodes]
    lengths = [int(e["length"]) for e in episodes]
    reasons = [e["termination_reason"] for e in episodes]

    avg_reward = sum(rewards) / len(rewards)
    avg_length = sum(lengths) / len(lengths)

    collision_count = sum(1 for r in reasons if "collision" in r)
    collision_rate = collision_count / len(reasons)

    timeout_count = sum(1 for r in reasons if r == "timeout")
    timeout_rate = timeout_count / len(reasons)

    stuck_count = sum(1 for r in reasons if r == "stuck")
    stuck_rate = stuck_count / len(reasons)

    oob_count = sum(1 for r in reasons if r == "out_of_bounds")
    oob_rate = oob_count / len(reasons)

    # Check for issues
    if collision_rate > 0.4:
        issues.append(f"High collision rate: {collision_rate:.0%}")
        actions.append({
            "type": "adjust",
            "param": "proximity_coeff",
            "value": -0.10,
            "reason": "Reduce proximity penalty to allow closer flying"
        })

    if avg_reward < 0 and len(episodes) > 20:
        issues.append(f"Negative avg reward: {avg_reward:.2f}")
        actions.append({
            "type": "adjust",
            "param": "progress_coeff",
            "value": 1.2,
            "reason": "Boost progress reward to encourage forward movement"
        })

    if stuck_rate > 0.3:
        issues.append(f"High stuck rate: {stuck_rate:.0%}")

    if oob_rate > 0.2:
        issues.append(f"High out-of-bounds rate: {oob_rate:.0%}")

    if avg_length < 50 and len(episodes) > 10:
        issues.append(f"Very short episodes: {avg_length:.0f} steps")

    # Check curriculum
    if curriculum:
        stage = curriculum.get("current_stage", 0)
        stage_steps = curriculum.get("stage_steps", 0)

        if stage_steps > 600_000:
            issues.append(f"Stage {stage} taking too long: {stage_steps:,} steps")

    return {
        "status": "ok" if not issues else "warning",
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "stuck_rate": stuck_rate,
        "oob_rate": oob_rate,
        "issues": issues,
        "actions": actions,
        "stage": curriculum.get("current_stage", 0) if curriculum else -1,
        "stage_steps": curriculum.get("stage_steps", 0) if curriculum else 0,
    }


def apply_actions(actions):
    """Apply recommended adjustments via control file."""
    if not actions:
        return

    control_file = config.LEARNING_DIR / "training_control.json"
    control = {
        "action": "adjust",
        "adjustments": {a["param"]: a["value"] for a in actions},
        "timestamp": time.time(),
        "reasons": [a["reason"] for a in actions],
    }
    with open(control_file, "w") as f:
        json.dump(control, f, indent=2)


def log_report(analysis):
    """Save monitoring report to file."""
    report_file = config.LEARNING_DIR / "monitor_report.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis": analysis,
    }
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)


def run_check():
    """Run a single monitoring check."""
    curriculum = load_curriculum()
    episodes = load_recent_episodes(50)
    analysis = analyze(episodes, curriculum)

    # Log report
    log_report(analysis)

    # Print summary
    print(f"[MONITOR] {time.strftime('%H:%M:%S')}")

    if analysis["status"] == "no_data":
        print("  Waiting for training data...")
        print("  (first checkpoint appears after ~20K steps)")
        return analysis

    print(f"  Stage: {analysis['stage']}/3, Steps: {analysis['stage_steps']:,}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Avg Reward: {analysis.get('avg_reward', 0):.2f}")
    print(f"  Avg Length: {analysis.get('avg_length', 0):.0f}")
    print(f"  Collision: {analysis.get('collision_rate', 0):.0%}")
    print(f"  Timeout: {analysis.get('timeout_rate', 0):.0%}")
    print(f"  Stuck: {analysis.get('stuck_rate', 0):.0%}")

    if analysis["issues"]:
        print(f"  ISSUES:")
        for issue in analysis["issues"]:
            print(f"    - {issue}")

    if analysis["actions"]:
        print(f"  ACTIONS:")
        for action in analysis["actions"]:
            print(f"    - {action['reason']}")
        apply_actions(analysis["actions"])

    return analysis


if __name__ == "__main__":
    import sys
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 900

    print(f"[MONITOR] Starting autonomous monitor (check every {interval}s)")
    while True:
        try:
            run_check()
        except Exception as e:
            print(f"[MONITOR] Error: {e}")
        time.sleep(interval)
