import csv
import json
import time
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
import config
from curriculum import CurriculumManager


class ConsoleMonitorCallback(BaseCallback):
    """Logs every episode to console + periodic summary."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reasons = []
        self.collision_count = 0
        self.completion_count = 0
        self.start_time = time.time()
        self.last_summary_ep = 0
        self.summary_interval = 10

        print(f"{'Time':>8}  {'Ep':>4}  {'Len':>4}  {'Reward':>7}  {'Reason'}")
        print("-" * 45)

    def _on_step(self):
        if self.locals.get("infos") is None:
            return True

        for info in self.locals["infos"]:
            if info.get("terminal_observation") is not None:
                ep_info = info.get("episode")
                if ep_info is not None:
                    r = ep_info["r"]
                    l = int(ep_info["l"])
                    reason = info.get("termination_reason", "?")

                    self.episode_rewards.append(r)
                    self.episode_lengths.append(l)
                    self.episode_reasons.append(reason)

                    ep_idx = len(self.episode_rewards)
                    print(f"{time.strftime('%H:%M:%S'):>8}  "
                          f"{ep_idx:>4d}  {l:>4d}  {r:+7.2f}  {reason}")

                if info.get("collision"):
                    self.collision_count += 1
                if info.get("termination_reason") == "completed_lap":
                    self.completion_count += 1

        current_ep = len(self.episode_rewards)
        if current_ep - self.last_summary_ep >= self.summary_interval:
            self.last_summary_ep = current_ep
            elapsed = time.time() - self.start_time
            n = max(len(self.episode_rewards[-20:]), 1)
            avg_r = sum(self.episode_rewards[-20:]) / n
            avg_l = sum(self.episode_lengths[-20:]) / n
            print(f"[SUM] Step {self.num_timesteps:_} | "
                  f"{current_ep}ep | avgR {avg_r:+.2f} | "
                  f"avgL {avg_l:.0f} | coll {self.collision_count} | "
                  f"done {self.completion_count} | "
                  f"{elapsed/60:.1f}min")
        return True


class CSVEpisodeLogger(BaseCallback):
    """Logs episode results to a CSV file."""

    def __init__(self, save_path=None):
        super().__init__(verbose=0)
        if save_path is None:
            save_path = config.LEARNING_DIR / "training_log.csv"
        self.save_path = Path(save_path)
        self.file_handle = None
        self.csv_writer = None
        self.episode_counter = 0
        self._init_file()

    def _init_file(self):
        self.file_handle = open(self.save_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.file_handle)
        self.csv_writer.writerow([
            "timestamp", "episode", "total_steps", "reward", "length",
            "collisions", "completions", "termination_reason",
        ])
        self.file_handle.flush()

    def _on_step(self):
        if self.locals.get("infos") is None:
            return True

        for info in self.locals["infos"]:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_counter += 1
                self.csv_writer.writerow([
                    time.strftime("%H:%M:%S"),
                    self.episode_counter,
                    self.num_timesteps,
                    f"{ep_info['r']:.3f}",
                    int(ep_info["l"]),
                    int(info.get("collision", 0)),
                    1 if info.get("termination_reason") == "completed_lap" else 0,
                    info.get("termination_reason", "unknown"),
                ])
                self.file_handle.flush()
        return True

    def close(self):
        if self.file_handle is not None:
            self.file_handle.close()


class CurriculumCallback(BaseCallback):
    """Periodically checks curriculum stage and logs transitions."""

    def __init__(self, check_interval=50_000, verbose=0):
        super().__init__(verbose)
        self.check_interval = check_interval
        self.curriculum = None
        self._base_env = None
        self.last_check = 0

    def _on_training_start(self):
        env = self.training_env
        while hasattr(env, "env"):
            env = env.env
        self._base_env = env
        if hasattr(env, "curriculum"):
            self.curriculum = env.curriculum
        # Update total_steps from loaded checkpoint
        if self._base_env and hasattr(self._base_env, "set_total_steps"):
            self._base_env.set_total_steps(self.num_timesteps)

    def _on_step(self):
        if self.curriculum is None:
            return True

        # Update total_steps every step
        if self._base_env and hasattr(self._base_env, "set_total_steps"):
            self._base_env.set_total_steps(self.num_timesteps)

        if self.num_timesteps - self.last_check < self.check_interval:
            return True

        self.last_check = self.num_timesteps

        status = self.curriculum.get_status()
        if self.verbose >= 1:
            print(self.curriculum.format_report())

        if self.curriculum.should_advance():
            result = self.curriculum.advance_stage()
            print(f"\n{'='*60}")
            print(f"[CURRICULUM] {result}")
            print(f"{'='*60}\n")

        elif self.curriculum.should_regress():
            result = self.curriculum.regress_stage()
            print(f"\n{'='*60}")
            print(f"[CURRICULUM] {result}")
            print(f"{'='*60}\n")

        return True


class ControlFileCallback(BaseCallback):
    """Reads training_control.json and applies parameter adjustments."""

    def __init__(self, check_interval=60_000, verbose=0):
        super().__init__(verbose)
        self.check_interval = check_interval
        self.last_check = 0
        self.control_file = config.LEARNING_DIR / "training_control.json"

    def _on_step(self):
        if self.num_timesteps - self.last_check < self.check_interval:
            return True

        self.last_check = self.num_timesteps

        if not self.control_file.exists():
            return True

        try:
            with open(self.control_file) as f:
                control = json.load(f)

            action = control.get("action")

            if action == "pause":
                print("\n[CONTROL] Pause signal received. Training paused.")
                print("[CONTROL] Delete training_control.json to resume.")
                while self.control_file.exists():
                    time.sleep(1.0)
                print("[CONTROL] Resumed.\n")

            elif action == "adjust":
                adjustments = control.get("adjustments", {})
                reasons = control.get("reasons", [])
                print(f"\n[CONTROL] Applying adjustments:")
                for param, value in adjustments.items():
                    print(f"  - {param}: {value}")
                for reason in reasons:
                    print(f"  Reason: {reason}")

                # Apply learning rate
                if "learning_rate" in adjustments:
                    new_lr = float(adjustments["learning_rate"])
                    self.model.learning_rate = new_lr
                    print(f"  [CONTROL] Learning rate set to {new_lr}")

                # Apply reward coefficient changes to config
                reward_params = {
                    "R_PROGRESS_COEFF": "progress_coeff",
                    "R_PROXIMITY_COEFF": "proximity_coeff",
                    "R_PROXIMITY_THRESHOLD": "proximity_threshold",
                    "R_STUCK": "stuck_penalty",
                    "R_SPEED_COEFF": "speed_coeff",
                    "R_SURVIVE": "survive_bonus",
                }
                for config_key, adj_key in reward_params.items():
                    if adj_key in adjustments:
                        new_val = float(adjustments[adj_key])
                        setattr(config, config_key, new_val)
                        print(f"  [CONTROL] {config_key} set to {new_val}")

                # Remove control file after applying
                self.control_file.unlink()
                print("[CONTROL] Adjustments applied.\n")

        except Exception as e:
            print(f"[CONTROL] Error reading control file: {e}")

        return True
