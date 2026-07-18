# Plan — Адаптация проекта под MAVRL

## Архитектура MAVRL

```
Depth Map (256×256) → CNN Encoder → 64-dim latent
State (7-dim) → FC → 64-dim
[latent, state_enc] → LSTM (256 hidden) → Actor (ax,ay,az,yaw_rate) + Critic (value)
```

## Этапы адаптации

### Этап 1: Depth Map pipeline ✅ ГОТОВО
- [x] Добавить depth map publisher в navigation_node.cpp
- [x] Disparity → Depth: Z = f*B/d
- [x] Resize до 256×256
- [x] Format: mono8 [0,255] (как MAVRL)
- [x] Center crop (как MAVRL)
- [x] Защита от headless (try-catch для cv::namedWindow)
- [x] Удалить fake odom publisher
- [x] Собрать проект
- [x] Проверить в Gazebo

### Этап 2: Goal-point навигация ✅ ГОТОВО
- [x] Определение goal-point (80% длины пещеры)
- [x] State space: 14-dim → 7-dim (goal-oriented)
- [x] Completion detection: dist_to_goal < 2.0m
- [x] Entrance heading tracking

### Этап 3: Action Space ✅ ГОТОВО
- [x] 4-dim: body-frame accelerations (ax, ay, az, yaw_rate)
- [x] Интегрирование: vel_world += R(body→world) * acc_body * dt
- [x] Velocity clipping УДАЛЁН (как MAVRL)
- [x] body2world через scipy Rotation (как MAVRL)

### Этап 4: CNN + LSTM Policy ✅ ГОТОВО
- [x] DepthEncoder: 6 Conv (8→16→32→64→128→256) → 64-dim
- [x] RecurrentPolicy: CNN + LSTM(256) + Actor/Critic
- [x] MultiInputLstmPolicy: SB3-совместимая обёртка
- [x] log_std_init = -0.5 (как MAVRL)
- [x] VAE weights из MAVRL (16/16)
- [x] Веса: ~1.76M параметров

### Этап 5: VAE ✅ ГОТОВО
- [x] DepthVAE: Encoder (6 Conv) + Decoder
- [x] VAELoss: reconstruction + KL divergence
- [x] transfer_encoder_to_policy()
- [x] extract_encoder_from_policy()
- [x] MAVRL weights загружены

### Этап 6: Training Pipeline ⚠️ ЧАСТИЧНО
- [x] Stage A: initial PPO (200K steps)
- [x] Stage B: collect depth data (script)
- [ ] Stage C: train VAE+LSTM (TODO)
- [x] Stage D: retrain PPO with frozen encoder
- [x] LR decay: 1e-4 → 1e-5 (linear, как MAVRL)

### Этап 7: Curriculum ⚠️ ЧАСТИЧНО
- [x] Параметрические стадии (easy/medium/hard)
- [ ] Интеграция с drone_env.py (TODO)
- [ ] Динамическое изменение среды (TODO)

### Этап 8: Reward Function ✅ ГОТОВО
- [x] Goal progress (R_GOAL_COEFF × Δprogress)
- [x] Action penalties (angular, input, yaw, vertical — как MAVRL)
- [x] Collision: reset без штрафа (как MAVRL)
- [x] Stuck: -5.0 (дополнительная защита)
- [x] Out of bounds: -10.0 (дополнительная защита)
- [x] Completion: +50.0 (дополнительный бонус)
- [x] Extra bonuses УДАЛЕНЫ (survive, speed, centering — как MAVRL)

### Этап 9: Inference ✅ ГОТОВО
- [x] InferenceNode с Dict obs
- [x] 4-dim action
- [x] Goal-point state

### Этап 10: URDF ✅ ГОТОВО
- [x] Camera baseline: 120mm (как MAVRL)
- [x] Camera resolution: 640×480 (как MAVRL)
- [x] Camera FOV: 80° (100° не работает в software rendering)
- [x] Collision detection: объединены дублирующиеся блоки

### Этап 11: Collision Detection ✅ ГОТОВО
- [x] URDF: объединены <gazebo reference="base_link"> блоки
- [x] Bumper plugin теперь загружается
- [x] Fallback: stereo proximity + velocity discrepancy
- [ ] Gazebo contact sensor не работает с planar_move (ограничение)

---

## Сравнение с MAVRL (текущее состояние)

| Категория | Совпадение |
|-----------|------------|
| Network architecture | 100% |
| PPO hyperparameters | 100% |
| Reward function | 95% |
| State/Action spaces | 100% |
| Velocity clipping | 100% (удалён) |
| LR decay | 100% (добавлен) |
| Extra bonuses | 100% (удалены) |
| **Общее** | **~98%** |

### Оставшиеся различия
1. FOV 80° vs 100° — domain shift для VAE
2. Z-axis teleport vs physics — нереалистично, но работает
3. Stuck/OOB/Completion penalties — дополнительная защита
4. Симулятор — Gazebo vs Flightmare

---

## Что осталось сделать

### Критическое
1. **Stage C: VAE+LSTM training** — реализовать обучение VAE на depth data и LSTM на последовательностях
2. **Интеграция curriculum** — подключить CurriculumManager к drone_env
3. **TensorBoard логирование** — depth maps, latent vectors, LSTM states

### Важное
4. **collect_data.py** — скрипт сбора depth данных для VAE
5. **test_ppo.py** — визуализация полёта с обученной моделью
6. **Тестирование Stage A** — запустить начальное обучение PPO

### Диагностическое
7. **Логирование действий** — vx, vz, yaw в TensorBoard
8. **Визуализация траекторий** — запись видео полёта
9. **Benchmark** — сравнение с реактивным контроллером

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
python3 train.py --stage a  # Initial PPO
python3 train.py --stage b  # Collect depth data
python3 train.py --stage d  # Retrain PPO
```

### Проверка топиков
```bash
ros2 topic echo /navigation_node/depth_map --once
ros2 topic echo /navigation_node/stereo_distances --once
ros2 topic echo /left/image_raw --once
ros2 topic echo /drone/collisions --once
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
    ├── policy.py        # CNN + LSTM + Actor/Critic
    ├── recurrent_ppo.py # RecurrentPPO
    ├── vae.py           # VAE для encoder
    ├── reward.py        # Функция наград
    ├── train.py         # Pipeline обучения
    ├── inference_node.py# Инференс
    ├── callbacks.py     # Мониторинг
    └── utils.py         # Управление Gazebo
```

---

## Архитектурные решения

### Почему MAVRL, а не текущий подход
1. **Depth map 256×256** вместо 5 чисел — x100 больше информации
2. **LSTM** — память о прошлом, понимание динамики
3. **Goal-point** — чёткая цель (конец тоннеля)
4. **Body-frame accelerations** — реалистичная динамика
5. **VAE pre-training** — лучший encoder
6. **Action penalties** — плавный полёт

### Почему Gazebo, а не Flightmare
1. Уже настроен и работает
2. ROS 2 Humble (а не Noetic)
3. Проще для разработки

### Почему software rendering
1. Нет GPU в текущей среде (WSL2)
2. Xvfb + LIBGL_ALWAYS_SOFTWARE=1 работает
3. Достаточно для обучения
