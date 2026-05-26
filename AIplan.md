# План реализации: Neural Network Control для Drone Offline Navigation

## Общая стратегия

**Цель:** Заменить реактивное уклонение в `control_node.py` на управление через обученную нейросеть (PPO + Behavior Cloning warm-start).

**Главное условие:** Существующий проект **не изменяется**. Все файлы в `src/`, `scripts/` остаются как есть. Всё новое — только в директории `learning/`.

**Метод:** BC (Behavior Cloning) → PPO fine-tuning, 1 000 000 шагов, смена пещеры каждые 5 эпизодов, headless Gazebo.

---

## Этап 1. Структура `learning/`

```
learning/
├── config.py                 # Все гиперпараметры и пути
├── utils.py                  # Управление Gazebo, сброс дрона, генерация пещеры
├── drone_env.py              # Gym-среда: Gazebo ↔ Stable-Baselines3
├── reward.py                 # Функция награды
├── callbacks.py              # Чекпоинты + мониторинг
├── collect_expert.py         # Сбор экспертных данных с текущего control_node
├── bc_model.py               # BC supervised обучение (PyTorch MLP)
├── train.py                  # BC → PPO пайплайн
├── inference_node.py         # ROS 2 нода для инференса обученной модели
├── launch/
│   └── inference_launch.py   # Запуск симуляции с NN вместо control_node
├── checkpoints/              # *.zip модели (gitignore)
├── tensorboard_logs/         # логи обучения (gitignore)
├── expert_data/              # *.npz записи эксперта (gitignore)
└── requirements.txt
```

---

## Этап 2. `config.py` — параметры

```python
# --- Симуляция ---
HEADLESS = True
CAVE_CHANGE_INTERVAL = 5           # новая пещера каждые 5 эпизодов
EPISODE_TIMEOUT_SEC = 300
DT = 0.05                          # 20Hz
SIM_STEP_TIMEOUT = 10              # сек на ожидание данных от сенсоров

# --- BC (Behavior Cloning) ---
BC_EXPERT_SAMPLES = 12_000         # ~10 мин экспертного полёта
BC_EPOCHS = 50
BC_LR = 1e-3
BC_BATCH_SIZE = 256
BC_HIDDEN = [64, 64]

# --- PPO ---
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
POLICY_KWARGS = {
    "net_arch": [64, 64],
    "activation_fn": "Tanh"
}

# --- Чекпоинты ---
CHECKPOINT_DIR = "learning/checkpoints/"
SAVE_FREQ = 20_000
LOG_DIR = "learning/tensorboard_logs/"
EXPERT_DIR = "learning/expert_data/"

# --- Награды ---
R_COLLISION = -10.0
R_COMPLETION = 50.0
R_STUCK_STEP = -0.5
R_FORWARD = 0.1
R_SURVIVE = 0.01
R_TIMEOUT = -5.0
```

---

## Этап 3. `utils.py` — управление симуляцией

Функции:

- `kill_gazebo()` — `pkill gzserver + gzclient`
- `generate_cave()` — вызов `scripts/procedural_cave.py` → новый `.world` файл
- `launch_gazebo(headless)` — запуск Gazebo с новым `.world`
- `wait_for_gazebo(timeout)` — ожидание `/gazebo/set_entity_state`
- `wait_for_topics(topics)` — ожидание `stereo_distances`, `odom`, `imu`
- `spawn_drone()` — `spawn_entity.py rescue_drone` на `z=1.0`
- `reset_drone()` — `set_entity_state` → `(0,0,1.0)`, обнуление скоростей
- `is_out_of_bounds(x, yaw, z)` — проверка вылета за пределы

**Логика смены пещеры:**
```
reset():
    if episode_count % CAVE_CHANGE_INTERVAL == 0:
        kill_gazebo()
        generate_cave()
        launch_gazebo(headless=HEADLESS)
        wait_for_gazebo()
    spawn_drone()
    wait_for_topics()
    episode_count += 1
else:
    reset_drone()
```

---

## Этап 4. `reward.py` — функция награды

```
compute_reward(obs, action, info, prev_obs):

    # 1. Прогресс вдоль оси тоннеля (signed_along_cave)
    if forward_progress > 0:
        reward += +0.1 * forward_progress
    elif forward_progress < -0.1:
        reward += -0.05

    # 2. Штраф за столкновение (любой тип)
    if collision:
        reward += -10.0
        terminated = True

    # 3. Штраф за застревание (скорость < 0.01 > 15c)
    if stuck_seconds > 15:
        reward += -5.0
        terminated = True

    # 4. Шаговая награда за выживание
    if not terminated:
        reward += +0.01

    # 5. Награда за полный цикл (тупик + возврат на старт)
    if completed_lap:
        reward += +50.0
        terminated = True

    # 6. Штраф за вылет за границы
    if out_of_bounds:
        reward += -3.0
        terminated = True

    # 7. Штраф за таймаут (300 сек)
    if elapsed > 300:
        reward += -5.0
        terminated = True

    return reward, terminated
```

---

## Этап 5. `drone_env.py` — Gym-среда

**Observation** (Box, 13 значений, нормализованы):

| № | Сигнал | Нормализация | Диапазон |
|---|--------|-------------|----------|
| 0-4 | stereo_distances[5] | `/10.0` | [0, 1.5] |
| 5 | x (относительно центра) | `/50.0` | [-1, 1] |
| 6 | y (относительно центра) | `/50.0` | [-1, 1] |
| 7 | z (высота) | `/3.5` | [0, 1] |
| 8-9 | sin(yaw), cos(yaw) | raw | [-1, 1] |
| 10 | vx (odom скорость) | raw | [-1, 1] |
| 11 | roll_rate | `/π` | [-1, 1] |
| 12 | pitch_rate | `/π` | [-1, 1] |

**Action** (Box(3), clipped):

| № | Команда | Диапазон |
|---|---------|----------|
| 0 | linear.x | [0.0, 0.8] (только вперёд) |
| 1 | linear.z | [-0.7, 0.7] |
| 2 | angular.z | [-1.2, 1.2] |

**Step:**
1. Применить `action` → `cmd_vel_pub.publish(twist)` + `_set_gazebo_z_velocity(vz)`
2. `time.sleep(DT)` (или spin + wait)
3. Собрать `obs` из callback-ов (буферизировать последние значения)
4. `reward, terminated = compute_reward(...)`
5. `truncated = (elapsed > 300)`
6. `return obs, reward, terminated, truncated, info`

**Reset:**
1. Если эпизод % 5 == 0: перезапустить симуляцию с новой пещерой
2. Иначе: `reset_drone()`
3. Очистить буферы наблюдений
4. Ждать первый валидный `obs`
5. `return obs, info`

---

## Этап 6. `collect_expert.py` — сбор данных эксперта

Запускается параллельно с существующей симуляцией (`simulation_launch.py`):

1. Подписаться на `/navigation_node/stereo_distances`, `/odom`, `/imu/data`, `/cmd_vel`
2. Буферизировать синхронные тройки `(obs, action)`
3. Сохранять каждые 10 шагов в `expert_data/batch_{timestamp}.npz`
4. Останов по достижении `BC_EXPERT_SAMPLES` (12 000)

Формат `.npz`:
```python
np.savez("batch.npz",
    observations=array(N, 13),
    actions=array(N, 3)
)
```

---

## Этап 7. `bc_model.py` — Behavior Cloning

**Архитектура MLP** (идентична PPO `MlpPolicy`):

```
Input(13) → Linear(64) → Tanh → Linear(64) → Tanh → Linear(3) → Output(action)
```

**Обучение:**

```
load_expert_data()            # загрузка всех .npz
dataset = TensorDataset(obs, actions)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

model = BCNet(13, 3, hidden=[64, 64])
optimizer = Adam(model, lr=1e-3)
mse_loss = MSELoss()

for epoch in range(50):
    for batch_obs, batch_act in dataloader:
        pred = model(batch_obs)
        loss = mse_loss(pred, batch_act)
        loss.backward()
        optimizer.step()
    log(epoch, loss)

torch.save(model.state_dict(), "expert_data/bc_policy.pt")
```

---

## Этап 8. `train.py` — BC → PPO пайплайн

```python
def main():
    # --- Фаза 1: BC (если нет чекпоинта PPO) ---
    if not resume_from_checkpoint:
        if not has_expert_data():
            logger.info("Phase 1a: Collecting expert data...")
            collect_expert_data()

        logger.info("Phase 1b: Training BC model...")
        bc_model = train_bc()                   # bc_policy.pt

        logger.info("Phase 1c: Creating PPO with BC warm-start...")
        env = DroneEnv()
        model = PPO("MlpPolicy", env, verbose=1, **ppo_kwargs)
        # Загружаем веса BC в policy сети PPO
        load_bc_into_ppo(bc_model, model.policy)
    else:
        env = DroneEnv()
        model = PPO.load(checkpoint_path, env=env)

    # --- Фаза 2: PPO fine-tuning ---
    callbacks = [
        CheckpointCallback(SAVE_FREQ, CHECKPOINT_DIR),
        ConsoleMonitorCallback(),
    ]

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=False
    )

    model.save(f"{CHECKPOINT_DIR}/final_model.zip")
```

**Загрузка BC в PPO:**

```python
def load_bc_into_ppo(bc_state_dict, ppo_policy):
    # BC features → PPO mlp_extractor
    # BC action_head → PPO action_net
    # PPO value_net — fresh init (не обучается BC)
    with torch.no_grad():
        ppo_policy.mlp_extractor.load_state_dict(bc_features_dict, strict=False)
        ppo_policy.action_net.load_state_dict(bc_action_dict, strict=False)
```

**Запуск:**

```bash
cd learning

# Сбор данных (если нужно):
python collect_expert.py             # параллельно ros2 launch simulation_launch.py

# Полный пайплайн (BC → PPO, всё в одном):
python train.py

# Продолжить с чекпоинта:
python train.py --resume checkpoints/rl_model_200000_steps.zip
```

---

## Этап 9. Мониторинг

**Console output** (каждые 5 эпизодов или 10k шагов):

```
[TRAIN] Ep 47 | Step 94500 | Reward +2.31 | Len 312 | Collisions 3 | Completions 0 | Cave 10/5
[PPO]   policy_loss=0.042 | value_loss=0.38 | entropy=0.89 | explained_variance=0.12
```

**TensorBoard:**

```
scalars:
  - rollout/ep_rew_mean
  - rollout/ep_len_mean
  - rollout/collision_rate     (custom)
  - rollout/completion_rate    (custom)
  - rollout/avg_forward_speed  (custom)
  - train/policy_loss
  - train/value_loss
  - train/entropy
  - time/total_timesteps
```

**Чекпоинты:** каждые 20 000 шагов в `learning/checkpoints/rl_model_{N}.zip`

**CSV-лог:** `learning/training_log.csv` (эпизод, шаги, награда, коллизии, completions, скорость)

---

## Этап 10. `inference_node.py` + `inference_launch.py`

**Нода инференса:**

```python
class InferenceNode(Node):
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
        # Подписки на /navigation_node/stereo_distances, /odom
        # Публикация /cmd_vel
        # Timer 20Hz: predict() → publish(cmd_vel)
```

**Launch-файл** — копия `simulation_launch.py` с заменой:

- `control_node` → `ExecuteProcess(cmd=['python3', 'learning/inference_node.py', '--model', model_path])`

Все остальные ноды (perception, navigation, Gazebo, RViz) — без изменений.

**Запуск:**

```bash
ros2 launch learning/launch/inference_launch.py
```

---

## Тайминг (оценка)

| Фаза | Длительность |
|------|-------------|
| BC data collection | ~10 мин (параллельно с симуляцией) |
| BC training | ~2 мин |
| PPO 1M steps @ 20Hz | ~13.9 ч симуляции |
| Headless Gazebo (×2-3) | ~5-7 ч |
| Cave resets (200 эпизодов / 5 = 40 × 17с) | ~0.2 ч |
| **Итого:** | **~5-8 ч** |

---

## Что НЕ меняется в существующем проекте

- `src/drone_control/drone_control/control_node.py` — **без изменений**
- `src/drone_navigation/src/navigation_node.cpp` — **без изменений**
- `src/drone_perception/src/perception_node.cpp` — **без изменений**
- `src/drone_simulation/launch/simulation_launch.py` — **без изменений**
- `src/drone_simulation/launch/spawn_drone.launch.py` — **без изменений**
- `src/drone_simulation/urdf/rescue_drone.urdf.xacro` — **без изменений**
- `scripts/procedural_cave.py` — **без изменений**
- `run_drone.sh` — **без изменений**
- `PROJECT_CONTEXT.md`, `INSPECTING_IMPROVEMENT_PLAN.md` — **без изменений**

Всё новое — только в `learning/`.
