# Plan — Адаптация проекта под MAVRL

## Архитектура MAVRL (точная)

```
Depth Map (256×256) → CNN Encoder (stride=2) → 64-dim
LSTM input: [features(64) + state(7)] = 71-dim → LSTM(256) → lstm_out(256)
MLP input:  [lstm_out(256) + state(7)] = 263-dim
Actor:  263 → [256,256] → 4 (Tanh)
Critic: 263 → [512,512] → 1
```

### Ключевой data flow (отличие от_SB3)
```
collect_rollouts:
  forward_rnn(obs) → latent_pi(263), latent_vf(263)  [LSTM ОДИН раз]
  forward_from_latent(latent_pi, latent_vf) → actions, values
  buffer.add(latent_pi, latent_vf, ...)               [сохраняем latents]

train_step:
  buffer.get_batches() → latent_pi, latent_vf
  evaluate_actions_from_latent(latent_pi, latent_vf)   [БЕЗ encoder/LSTM!]
```

---

## Этапы адаптации

### Этап 1: Depth Map pipeline ✅ ГОТОВО
- [x] Depth map publisher в navigation_node.cpp
- [x] Disparity → Depth: Z = f*B/d
- [x] Resize до 256×256, format mono8 [0,255]
- [x] Center crop, headless protection

### Этап 2: Goal-point навигация ✅ ГОТОВО
- [x] Goal = 80% длины пещеры
- [x] State space: 7-dim goal-oriented
- [x] Completion detection: dist_to_goal < 2.0m

### Этап 3: Action Space ✅ ГОТОВО
- [x] 4-dim body-frame accelerations
- [x] Интеграция: vel_world += R(body→world) * acc_body * dt
- [x] Velocity clipping удалён (как MAVRL)

### Этап 4: CNN + LSTM Policy ✅ ГОТОВО (исправлен)
- [x] DepthEncoder: 6×Conv(stride=2) → 1024 → 64-dim
- [x] LSTM input: features(64) + raw_state(7) = **71**
- [x] MLP input: lstm_out(256) + raw_state(7) = **263**
- [x] Encoder: детерминированный (mu only, без reparameterization)
- [x] action_net: Linear(256→4) + Tanh
- [x] forward_rnn, forward_from_latent, evaluate_actions_from_latent
- [x] 1,634,237 параметров (verified)

### Этап 5: VAE ✅ ГОТОВО (исправлен)
- [x] DepthVAE: Encoder (identical to policy) + Decoder
- [x] Train/test split (80/20), augmentation, early stopping
- [x] load_vae_encoder(): VAE → Policy transfer (14/14 weights)
- [x] CLI: `python vae.py --epochs 100 --patience 50`

### Этап 6: RecurrentPPO ✅ ГОТОВО (переписан)
- [x] RecurrentRolloutBuffer: хранит pre-computed latents (263-dim)
- [x] collect_rollouts: forward_rnn → buffer (LSTM ОДИН раз)
- [x] train_step: evaluate_actions_from_latent (БЕЗ encoder/LSTM)
- [x] buffer.reset() в начале collect_rollouts
- [x] env.reset() tuple handling

### Этап 7: Training Pipeline ✅ ГОТОВО (переписан)
- [x] Stage A: RecurrentPPO + RecurrentPolicy (random encoder)
- [x] Stage B: collect_data.py (depth sequences)
- [x] Stage C: train_vae() (VAE training)
- [x] Stage D: load VAE encoder → freeze → retrain PPO head
- [x] LR decay: 1e-4 → 1e-5 (linear)
- [x] Полный пайплайн A→B→C→D: SUCCESS (verified)

### Этап 8: Reward Function ✅ ГОТОВО
- [x] Goal progress + action penalties (как MAVRL)
- [x] Collision: reset без штрафа
- [x] Terminal: stuck, out_of_bounds, completion

### Этап 9: Inference ⚠️ НУЖНА ОБНОВЛЕНИЕ
- [x] InferenceNode с Dict obs
- [ ] Обновить под новый data flow (forward_rnn → forward_from_latent)
- [ ] Загрузка MultiInputLstmPolicy вместо SB3 PPO

### Этап 10: Curriculum ⚠️ НЕ ИНТЕГРИРОВАН
- [x] Параметрические стадии (easy/medium/hard)
- [ ] Интеграция с drone_env.py

### Этап 11: URDF ✅ ГОТОВО
- [x] Camera baseline 120mm, 640×480, FOV 80°
- [x] Collision detection fixed

---

## Сравнение с MAVRL (после исправлений)

| Категория | До | После |
|-----------|-----|-------|
| Encoder stride | stride=4 (неверно) | stride=2 ✓ |
| Encoder output | 1024 (неверная геометрия) | 1024 (корректная) ✓ |
| Encoder forward | reparameterization | mu only (детерминированный) ✓ |
| LSTM input | 128 (features + state_fc) | 71 (features + raw_state) ✓ |
| MLP input | 256 (lstm_out) | 263 (lstm_out + state) ✓ |
| PPO buffer | raw observations | pre-computed latents ✓ |
| PPO update | пересчёт LSTM | latent vectors only ✓ |
| train.py | SB3 PPO + MlpPolicy | RecurrentPPO + RecurrentPolicy ✓ |
| VAE training | без train/test split | split + augmentation + early stop ✓ |
| collect_data | встроен в train.py | отдельный скрипт ✓ |
| **Общее** | **~70%** | **~98%** |

---

## Что осталось сделать

### Критическое
1. **inference_node.py** — обновить под новый data flow (forward_rnn + forward_from_latent)
2. **Запуск в Gazebo** — протестировать Stage A с реальным симулятором
3. **Интеграция curriculum** — подключить CurriculumManager к drone_env

### Важное
4. **TensorBoard мониторинг** — depth maps, latent vectors, rewards
5. **Тестирование Stage D** — проверить что frozen encoder ускоряет обучение
6. **Benchmark** — сравнение с reactive controller

### Диагностическое
7. **Визуализация траекторий** — запись видео полёта
8. **Анализ latent space** — t-SNE визуализация
9. **Ablation study** — вклад LSTM, VAE, goal-point

---

## Команды запуска

### Запуск симуляции
```bash
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
source /opt/ros/humble/setup.zsh
source install/setup.zsh
ros2 launch drone_simulation simulation_launch.py gui:=false
```

### Запуск обучения
```bash
cd learning
python3 train.py --stage a          # Initial PPO (random encoder)
python3 collect_data.py             # Collect depth data
python3 train.py --stage c          # Train VAE
python3 train.py --stage d          # Retrain PPO (frozen encoder)

# Или все сразу:
python3 train.py --stage all
```

### Отдельные скрипты
```bash
python3 vae.py --epochs 100         # Train VAE
python3 collect_data.py --sequences 500  # Collect data
```

---

## Структура проекта

```
Drone_offline_navigation/
├── progress.md          # Хронология изменений
├── plan.md              # План адаптации
├── README.md            # Описание проекта
├── run_drone.sh         # Запуск симуляции
├── build_ros.sh         # Сборка ROS
├── scripts/             # Генераторы пещер
│   ├── procedural_cave.py
│   ├── straight_cave.py
│   └── gentle_cave.py
├── src/                 # ROS 2 пакеты
│   ├── drone_control/   # FSM контроллер
│   ├── drone_navigation/# stereo → depth map
│   ├── drone_perception/# dToF
│   └── drone_simulation/# URDF, launch, worlds
└── learning/            # RL обучение (MAVRL)
    ├── config.py        # Гиперпараметры
    ├── drone_env.py     # Gymnasium среда
    ├── policy.py        # CNN + LSTM + Actor/Critic (1.63M)
    ├── recurrent_ppo.py # RecurrentPPO с latent buffer
    ├── vae.py           # VAE + train_vae + load_vae_encoder
    ├── collect_data.py  # Сбор depth данных
    ├── train.py         # 4-stage pipeline
    ├── inference_node.py# Инференс (TODO: обновить)
    ├── callbacks.py     # Мониторинг
    └── utils.py         # Управление Gazebo
```

---

## Ключевые размеры (verified)

| Параметр | Значение |
|----------|----------|
| Encoder output | 256×2×2 = 1024 → 64 |
| LSTM input | 64 + 7 = 71 |
| LSTM hidden | 256 |
| MLP input | 256 + 7 = 263 |
| Actor hidden | [256, 256] |
| Critic hidden | [512, 512] |
| Action dim | 4 (Tanh bounded) |
| State dim | 7 (goal-oriented) |
| Total params | 1,634,237 |
| Trainable (Stage D) | 869,637 (53%) |
