"""
Curriculum Learning Manager for drone training.
Manages progression through training stages with increasing difficulty.
"""

import json
from pathlib import Path
import config


class CurriculumManager:
    """Manages curriculum learning stages for drone training."""

    def __init__(self):
        self.stages = config.CURRICULUM_STAGES
        self.current_stage = 0
        self.stage_steps = 0
        self.total_steps = 0
        self.episode_results = []  # (success, length, reward)
        self.stage_start_step = 0
        self.metrics_file = config.LEARNING_DIR / "curriculum_metrics.json"
        self._load_metrics()

    def _load_metrics(self):
        """Load saved metrics if they exist."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r") as f:
                    data = json.load(f)
                    self.current_stage = data.get("current_stage", 0)
                    self.stage_steps = data.get("stage_steps", 0)
                    self.total_steps = data.get("total_steps", 0)
                    self.episode_results = data.get("episode_results", [])
                    self.stage_start_step = data.get("stage_start_step", 0)
            except Exception:
                pass

    def _save_metrics(self):
        """Save metrics to file."""
        data = {
            "current_stage": self.current_stage,
            "stage_steps": self.stage_steps,
            "total_steps": self.total_steps,
            "episode_results": self.episode_results[-1000:],  # Keep last 1000
            "stage_start_step": self.stage_start_step,
        }
        with open(self.metrics_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_current_stage(self):
        """Get current curriculum stage config."""
        return self.stages[self.current_stage]

    def get_cave_script(self):
        """Get path to current stage's cave generator script."""
        stage = self.get_current_stage()
        script_name = stage["cave_script"]
        if script_name == "procedural_cave.py":
            return config.SCRIPTS_DIR / script_name
        else:
            return config.SCRIPTS_DIR / script_name

    def get_reward_coefficients(self):
        """Get reward coefficients for current stage."""
        stage = self.get_current_stage()
        return {
            "progress_coeff": stage["progress_coeff"],
            "proximity_coeff": stage["proximity_coeff"],
            "proximity_threshold": stage.get("proximity_threshold", 0.5),
        }

    def record_episode(self, success, length, reward, termination_reason=""):
        """Record an episode result."""
        self.episode_results.append({
            "success": success,
            "length": length,
            "reward": reward,
            "termination_reason": termination_reason,
            "stage": self.current_stage,
            "total_steps": self.total_steps,
        })
        self._save_metrics()

    def update_steps(self, steps):
        """Update total steps and stage steps."""
        self.total_steps = steps
        self.stage_steps = steps - self.stage_start_step
        self._save_metrics()

    def success_rate(self, window=100):
        """Calculate success rate over last N episodes."""
        if not self.episode_results:
            return 0.0

        stage_episodes = [
            r for r in self.episode_results
            if r["stage"] == self.current_stage
        ]

        if len(stage_episodes) < 10:
            return 0.0

        recent = stage_episodes[-window:]
        successes = sum(1 for r in recent if r["success"])
        return successes / len(recent)

    def avg_reward(self, window=100):
        """Calculate average reward over last N episodes."""
        if not self.episode_results:
            return 0.0

        stage_episodes = [
            r for r in self.episode_results
            if r["stage"] == self.current_stage
        ]

        if not stage_episodes:
            return 0.0

        recent = stage_episodes[-window:]
        return sum(r["reward"] for r in recent) / len(recent)

    def avg_length(self, window=100):
        """Calculate average episode length over last N episodes."""
        if not self.episode_results:
            return 0.0

        stage_episodes = [
            r for r in self.episode_results
            if r["stage"] == self.current_stage
        ]

        if not stage_episodes:
            return 0.0

        recent = stage_episodes[-window:]
        return sum(r["length"] for r in recent) / len(recent)

    def collision_rate(self, window=100):
        """Calculate collision rate over last N episodes."""
        if not self.episode_results:
            return 0.0

        stage_episodes = [
            r for r in self.episode_results
            if r["stage"] == self.current_stage
        ]

        if len(stage_episodes) < 10:
            return 0.0

        recent = stage_episodes[-window:]
        collisions = sum(1 for r in recent if r.get("termination_reason", "").startswith("collision"))
        return collisions / len(recent)

    def should_advance(self):
        """Check if we should advance to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False

        if self.stage_steps < config.CURRICULUM_STAGE_STEPS:
            return False

        rate = self.success_rate()
        return rate > config.CURRICULUM_ADVANCE_THRESHOLD

    def should_regress(self):
        """Check if we should go back to previous stage."""
        if self.current_stage <= 0:
            return False

        if self.stage_steps < 200_000:
            return False

        rate = self.success_rate()
        return rate < config.CURRICULUM_REGRESS_THRESHOLD

    def advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            old_stage = self.stages[self.current_stage]["name"]
            self.current_stage += 1
            self.stage_steps = 0
            self.stage_start_step = self.total_steps
            new_stage = self.stages[self.current_stage]["name"]
            self._save_metrics()
            return f"ADVANCED: {old_stage} → {new_stage}"
        return "ALREADY_AT_MAX_STAGE"

    def regress_stage(self):
        """Go back to previous curriculum stage."""
        if self.current_stage > 0:
            old_stage = self.stages[self.current_stage]["name"]
            self.current_stage -= 1
            self.stage_steps = 0
            self.stage_start_step = self.total_steps
            new_stage = self.stages[self.current_stage]["name"]
            self._save_metrics()
            return f"REGRESSED: {old_stage} → {new_stage}"
        return "ALREADY_AT_MIN_STAGE"

    def get_status(self):
        """Get current curriculum status."""
        stage = self.get_current_stage()
        return {
            "stage": self.current_stage + 1,
            "total_stages": len(self.stages),
            "stage_name": stage["name"],
            "stage_steps": self.stage_steps,
            "total_steps": self.total_steps,
            "success_rate": self.success_rate(),
            "avg_reward": self.avg_reward(),
            "avg_length": self.avg_length(),
            "collision_rate": self.collision_rate(),
            "should_advance": self.should_advance(),
            "should_regress": self.should_regress(),
        }

    def format_report(self):
        """Format a training report."""
        status = self.get_status()
        return (
            f"[CURRICULUM] Stage {status['stage']}/{status['total_stages']} ({status['stage_name']})\n"
            f"  Steps: {status['stage_steps']:,} / {config.CURRICULUM_STAGE_STEPS:,}\n"
            f"  Success Rate: {status['success_rate']:.1%}\n"
            f"  Avg Reward: {status['avg_reward']:.1f}\n"
            f"  Avg Length: {status['avg_length']:.0f} steps\n"
            f"  Collision Rate: {status['collision_rate']:.1%}\n"
            f"  Should Advance: {status['should_advance']}\n"
            f"  Should Regress: {status['should_regress']}"
        )
