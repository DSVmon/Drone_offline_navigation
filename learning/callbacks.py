import csv
import time
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
import config


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
                    ep_num = len(self.locals["infos"])  # not reliable

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
