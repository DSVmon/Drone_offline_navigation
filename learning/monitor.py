#!/usr/bin/env python3
"""
Training monitor and controller for drone RL training.
Provides real-time monitoring, parameter adjustment, and control.
"""

import argparse
import json
import time
from pathlib import Path
import sys

import config


def load_metrics():
    """Load training metrics from files."""
    metrics = {}

    # Load curriculum metrics
    curriculum_file = config.LEARNING_DIR / "curriculum_metrics.json"
    if curriculum_file.exists():
        with open(curriculum_file, "r") as f:
            metrics["curriculum"] = json.load(f)

    # Load training log
    log_file = config.LEARNING_DIR / "training_log.csv"
    if log_file.exists():
        episodes = []
        with open(log_file, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 8:
                        episodes.append({
                            "timestamp": parts[0],
                            "episode": int(parts[1]),
                            "total_steps": int(parts[2]),
                            "reward": float(parts[3]),
                            "length": int(parts[4]),
                            "collisions": int(parts[5]),
                            "completions": int(parts[6]),
                            "termination_reason": parts[7],
                        })
        metrics["episodes"] = episodes

    # Load TensorBoard logs (latest)
    tb_dir = config.LOG_DIR
    if tb_dir.exists():
        tb_files = list(tb_dir.glob("**/*.tensors"))
        if tb_files:
            metrics["tensorboard_files"] = len(tb_files)

    return metrics


def show_status():
    """Show current training status."""
    metrics = load_metrics()

    print("=" * 60)
    print("TRAINING STATUS")
    print("=" * 60)

    if "curriculum" in metrics:
        c = metrics["curriculum"]
        stage_names = ["straight", "gentle", "full"]
        stage_idx = c.get("current_stage", 0)
        stage_name = stage_names[stage_idx] if stage_idx < len(stage_names) else "unknown"
        print(f"Stage: {stage_idx + 1}/3 ({stage_name})")
        print(f"Stage Steps: {c.get('stage_steps', 0):,}")
        print(f"Total Steps: {c.get('total_steps', 0):,}")

        # Calculate success rate from episode_results
        episodes = c.get("episode_results", [])
        stage_eps = [e for e in episodes if e.get("stage") == stage_idx]
        if len(stage_eps) >= 10:
            recent = stage_eps[-100:]
            successes = sum(1 for e in recent if e.get("success"))
            success_rate = successes / len(recent)
            avg_reward = sum(e.get("reward", 0) for e in recent) / len(recent)
            avg_length = sum(e.get("length", 0) for e in recent) / len(recent)
            print(f"  Success Rate: {success_rate:.1%}")
            print(f"  Avg Reward: {avg_reward:.1f}")
            print(f"  Avg Length: {avg_length:.0f}")
        else:
            print(f"  Episodes on stage: {len(stage_eps)} (need 10+ for stats)")
    else:
        print("No curriculum data found")

    if "episodes" in metrics and metrics["episodes"]:
        episodes = metrics["episodes"]
        print(f"\nEpisodes: {len(episodes)}")

        # Last 100 episodes stats
        recent = episodes[-100:]
        avg_reward = sum(e["reward"] for e in recent) / len(recent)
        avg_length = sum(e["length"] for e in recent) / len(recent)

        # Termination reasons
        reasons = {}
        for e in recent:
            r = e["termination_reason"]
            reasons[r] = reasons.get(r, 0) + 1

        print(f"Last 100 episodes:")
        print(f"  Avg Reward: {avg_reward:.1f}")
        print(f"  Avg Length: {avg_length:.0f}")
        print(f"  Termination Reasons:")
        for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {r}: {count} ({count/len(recent)*100:.0f}%)")
    else:
        print("\nNo episode data found")

    if "tensorboard_files" in metrics:
        print(f"\nTensorBoard logs: {metrics['tensorboard_files']} files")

    print("=" * 60)


def pause_training():
    """Pause training by creating a control file."""
    control_file = config.LEARNING_DIR / "training_control.json"
    control = {"action": "pause", "timestamp": time.time()}
    with open(control_file, "w") as f:
        json.dump(control, f, indent=2)
    print("[MONITOR] Pause signal sent")


def resume_training():
    """Resume training by removing the control file."""
    control_file = config.LEARNING_DIR / "training_control.json"
    if control_file.exists():
        control_file.unlink()
        print("[MONITOR] Resume signal sent")
    else:
        print("[MONITOR] No pause signal found")


def adjust_parameters(**kwargs):
    """Adjust training parameters."""
    control_file = config.LEARNING_DIR / "training_control.json"
    control = {
        "action": "adjust",
        "adjustments": kwargs,
        "timestamp": time.time(),
    }
    with open(control_file, "w") as f:
        json.dump(control, f, indent=2)
    print(f"[MONITOR] Adjustments: {kwargs}")


def show_help():
    """Show help message."""
    print("""
Drone Training Monitor

Usage:
    python3 learning/monitor.py [command]

Commands:
    status          Show current training status
    pause           Pause training
    resume          Resume training
    adjust KEY=VAL  Adjust parameters (e.g., learning_rate=5e-5)
    help            Show this help message

Examples:
    python3 learning/monitor.py status
    python3 learning/monitor.py pause
    python3 learning/monitor.py resume
    python3 learning/monitor.py adjust learning_rate=5e-5 progress_coeff=0.6
""")


def main():
    parser = argparse.ArgumentParser(description="Drone Training Monitor")
    parser.add_argument("command", nargs="?", default="status", help="Command to execute")
    parser.add_argument("args", nargs="*", help="Additional arguments")

    args = parser.parse_args()

    if args.command == "status":
        show_status()
    elif args.command == "pause":
        pause_training()
    elif args.command == "resume":
        resume_training()
    elif args.command == "adjust":
        if not args.args:
            print("Error: No parameters specified")
            print("Example: python3 learning/monitor.py adjust learning_rate=5e-5")
            sys.exit(1)
        params = {}
        for arg in args.args:
            if "=" in arg:
                key, val = arg.split("=", 1)
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
        adjust_parameters(**params)
    elif args.command == "help":
        show_help()
    else:
        print(f"Unknown command: {args.command}")
        show_help()


if __name__ == "__main__":
    main()
