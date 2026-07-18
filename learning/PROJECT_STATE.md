# Drone Offline Navigation — Полное техническое описание проекта

## 1. Обзор проекта

**Цель:** Разработка автономной системы навигации спасательного дрона для исследования закрытых пространств (шахт, пещер) без GPS.

**Метод:** Стереозрение (OpenCV StereoBM) как основной сенсор для 5-зонного анализа глубины и реактивного 3D-управления.

**Фазы проекта:**
- **Phase 4.2:** Реактивное управление в Gazebo Classic (текущая рабочая версия)
- **Phase 4.3:** Нейросеть BC→PPO в `learning/` (в процессе обучения)

---

## 2. Технический стек

| Компонент | Технология |
|-----------|------------|
| ОС | Ubuntu 22.04 (WSL2) |
| Middleware | ROS 2 Humble |
| Симулятор | Gazebo Classic |
| Языки | C++ (навигация/перцепция), Python (управление/обучение) |
| Компьютерное зрение | OpenCV 4.x, cv_bridge, image_geometry, message_filters |
| ML фреймворки | PyTorch, Stable-Baselines3, Gymnasium |
| Визуализация | RViz2, TensorBoard |
| Контроль версий | Git |

---

## 3. Архитектура системы

### 3.1. ROS 2 ноды (Phase 4.2)

```
┌─────────────────────────────────────────────────────────────┐
│                      Gazebo Classic                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ Camera  │  │ Camera  │  │  IMU    │  │  Odometry│      │
│  │  Left   │  │  Right  │  │         │  │          │      │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬─────┘      │
└───────┼─────────────┼────────────┼────────────┼────────────┘
        │             │            │            │
   /left/image_raw  /right/image_raw  /imu/data  /odom
        │             │            │            │
        ▼             ▼            ▼            │
┌───────────────────────────────────┐           │
│      navigation_node (C++)        │           │
│  StereoBM → Disparity → 5 zones  │           │
│  Компенсация наклона (IMU)        │           │
└───────────────┬───────────────────┘           │
                │                               │
                ▼                               │
    /navigation_node/stereo_distances           │
                │                               │
                ▼                               ▼
┌───────────────────────────────────────────────────────────┐
│                  control_node (Python)                     │
│  Конечный автомат: SEARCHING → INSPECTING → TURNING       │
│  P-контроллер разворота, 3D-руление                       │
│  Гибридное управление: X/Y через planar_move, Z через    │
│  сервис /gazebo/set_entity_state                          │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
                       /cmd_vel
                            │
                            ▼
                       Gazebo (движение дрона)
```

### 3.2. ROS 2 ноды (Phase 4.3 — обучение)

```
┌─────────────────────────────────────────────────────────────┐
│                      Gazebo Classic                         │
│  (headless, без GUI)                                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
        /navigation_node/stereo_distances
        /odom
        /drone/collisions
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  drone_env.py (Gymnasium)                   │
│  Observation: 14-dim (stereo[5] + позиция + скорость)      │
│  Action: 3-dim (vx, vz, yaw)                               │
│  Reward: progress + centering + altitude + collision        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  PPO (Stable-Baselines3)                    │
│  Network: [256, 256] ReLU, ~70K параметров                 │
│  2M шагов, curriculum learning                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
                   model.zip (чекпоинт)
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  inference_node.py (ROS 2)                  │
│  Загружает модель, pub cmd_vel                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Структура файлов

### 4.1. Исходный код (src/)

| Файл | Строк | Назначение |
|------|-------|------------|
| `drone_control/control_node.py` | 1124 | Конечный автомат миссии, управление дроном |
| `drone_navigation/src/navigation_node.cpp` | 297 | Обработка стереопары, расчёт 5 зон |
| `drone_perception/src/perception_node.cpp` | 65 | Мониторинг dToF дальномеров (отключён) |
| `drone_simulation/launch/simulation_launch.py` | 90 | Запуск симуляции |
| `drone_simulation/urdf/rescue_drone.urdf.xacro` | — | Модель дрона |

### 4.2. Обучение (learning/)

| Файл | Строк | Назначение |
|------|-------|------------|
| `config.py` | 132 | Гиперпараметры, пути, награды |
| `drone_env.py` | 579 | Gymnasium среда (ROS 2 ↔ PPO) |
| `reward.py` | 207 | Функция наград (10+ компонентов) |
| `curriculum.py` | 231 | Управление стадиями обучения |
| `callbacks.py` | 236 | ConsoleMonitor, CSVEpisodeLogger, CurriculumCallback, ControlFileCallback |
| `train.py` | 181 | BC → PPO пайплайн |
| `bc_model.py` | 229 | Behavior Cloning (PyTorch MLP) |
| `inference_node.py` | 193 | ROS 2 инференс |
| `utils.py` | 166 | Управление Gazebo |
| `collect_expert.py` | 192 | Сбор экспертных данных |
| `auto_monitor.py` | 191 | Автономный мониторинг |
| `monitor.py` | 212 | Ручной мониторинг |
| `diagnostic.py` | 102 | Диагностика |

### 4.3. Скрипты (scripts/)

| Файл | Строк | Назначение |
|------|-------|------------|
| `procedural_cave.py` | 317 | Stage 2: сложная пещера 100м |
| `straight_cave.py` | 242 | Stage 0: прямой туннель 100м + препятствия |
| `gentle_cave.py` | 246 | Stage 1: извилистый туннель 100м ±30° |

---

## 5. Observation Space (14 параметров)

| Индекс | Параметр | Нормализация | Диапазон | Описание |
|--------|----------|--------------|----------|----------|
| 0 | left stereo | /10.0 | [0, 1.5] | Расстояние слева |
| 1 | center stereo | /10.0 | [0, 1.5] | Расстояние впереди |
| 2 | right stereo | /10.0 | [0, 1.5] | Расстояние справа |
| 3 | top stereo | /10.0 | [0, 1.5] | Расстояние сверху |
| 4 | bottom stereo | /10.0 | [0, 1.5] | Расстояние снизу |
| 5 | x | /50.0 | [-1, 1] | Позиция по x |
| 6 | y | /50.0 | [-1, 1] | Позиция по y |
| 7 | z | /3.5 | [0, 1] | Высота |
| 8 | sin(yaw) | raw | [-1, 1] | Угол поворота (sin) |
| 9 | cos(yaw) | raw | [-1, 1] | Угол поворота (cos) |
| 10 | vx | clip | [-1, 1] | Скорость вперёд |
| 11 | vz | clip | [-1, 1] | Скорость вверх |
| 12 | roll/π | /π | [-1, 1] | Крен |
| 13 | pitch/π | /π | [-1, 1] | Тангаж |

---

## 6. Action Space (3 параметра)

| Индекс | Параметр | Физический диапазон | Назначение |
|--------|----------|---------------------|------------|
| 0 | vx | [0, 1.5] м/с | Движение вперёд |
| 1 | vz | [-0.5, 0.5] м/с | Движение вверх/вниз |
| 2 | yaw | [-1.0, 1.0] рад/с | Поворот |

---

## 7. Network Architecture

```
Input(14) → Linear(256) → ReLU → Linear(256) → ReLU → Output(3)
```

**Параметров:** ~70K

---

## 8. Функция наград

### 8.1. Положительные (за правильное поведение)

| # | Компонент | Формула | Макс/шаг | Описание |
|---|-----------|---------|----------|----------|
| 1 | Progress | progress_coeff × Δx × 0.8 | +0.024 | Лети вперёд |
| 2 | Survival | const | +0.01 | Живи |
| 2b | Altitude (z=1.75м) | | +0.15 | Лети на правильной высоте |
| 3 | Speed | speed_coeff × vx × 0.3 | +0.007 | Лети быстро |
| 4b | Turn near wall | | +0.15 | Поворачивайся у стен |
| 4c | Exploration | | +0.03 | Сканируй окружение |
| 4d | Dodge (d растёт) | | +0.10 | Уворачивайся |
| 6 | Centering (horizontal) | 0.20 × balance | +0.20 | Лети по центру |
| 7 | Centering (vertical) | 0.20 × balance | +0.20 | Лети по центру |
| 7c | Dead-end turn | | +0.30 | Разворачивайся от тупика |

### 8.2. Отрицательные (за неправильное поведение)

| # | Компонент | Минимум/шаг | Описание |
|---|-----------|-------------|----------|
| 2b | Altitude z<0.8м | -0.50 | Слишком низко (камера сломана) |
| 2c | Floor bottom<0.2м | -0.80 | Очень низко (камера слепа) |
| 3b | Backward flight | -0.05 | Летишь назад |
| 4b | No turn near wall | -0.02 | Не поворачиваешься у стены |
| 4d | Approaching wall | -0.02 | Приближаешься к стене |
| 7b | Near ANY wall (min<0.5м) | -0.05 | Близко к любой поверхности |
| 7d | Repeat flight path | -0.10 | Копируешь маршрут |

### 8.3. Terminal penalties

| Событие | Штраф | Описание |
|---------|-------|----------|
| Collision | -15.0 | Реальный контакт (Gazebo sensor) |
| Stuck | -5.0 | 3 сек без движения |
| Out of bounds | -10.0 | Вылет за границы |
| Timeout | -3.0 | 600 сек |
| Completed lap | +50.0 | Полный цикл |

---

## 9. Curriculum Learning

### 9.1. Стадии

| Stage | Пещера | Длина | Ширина | Повороты | Препятствия |
|-------|--------|-------|--------|----------|-------------|
| 0 (straight) | straight_cave.py | 100м | 6м | нет | Балки + колонны |
| 1 (gentle) | gentle_cave.py | 100м | 5м | ±30° | Сталактиты |
| 2 (full) | procedural_cave.py | 100м | 4.5м | ±55° | Сталактиты + плиты |

### 9.2. Параметры curriculum

| Параметр | Значение |
|----------|----------|
| CAVE_CHANGE_INTERVAL | 20 эпизодов |
| CURRICULUM_STAGE_STEPS | 500K шагов |
| CURRICULUM_ADVANCE_THRESHOLD | 80% success |
| CURRICULUM_REGRESS_THRESHOLD | 30% success |

### 9.3. Награды по стадиям

| Stage | progress_coeff | proximity_coeff |
|-------|----------------|-----------------|
| 0 (straight) | 1.0 | -0.05 |
| 1 (gentle) | 0.7 | -0.10 |
| 2 (full) | 0.5 | -0.15 |

---

## 10. Текущее состояние обучения

| Параметр | Значение |
|----------|----------|
| Текущий stage | 1 (gentle cave) |
| Текущий чекпоинт | 780K steps |
| Всего шагов | ~820K |
| Всего эпизодов | ~900 |
| Reward trend | Стабилизируется |
| Out of bounds | ~70-80% |
| Collision | 0% |
| Stuck | ~15% |

---

## 11. Известные проблемы

1. **OBS_POS_MAX=50 при BOUNDS_XY=110** — observations для x>50 выходят за bounds. PPO клippит, адаптируется за 50-100K шагов.

2. **Training divergence при переходе на Stage 1** — модель деградировала при смене пещеры. Текущий 780K чекпоинт может быть нестабильным.

3. **Virtual collision удалён** — раньше drone получал collision при d<0.25м. Сейчас только реальный collision через Gazebo contact sensor.

---

## 12. Как запускать

### Обучение
```bash
cd learning && python3 train.py --resume checkpoints/rl_model_780000_steps.zip
```

### Мониторинг
```bash
cd learning && python3 auto_monitor.py 600
```

### Диагностика
```bash
cd learning && python3 diagnostic.py
```

### TensorBoard
```bash
source learning/venv/bin/activate
tensorboard --logdir learning/tensorboard_logs
```

### Инференс
```bash
ros2 launch learning/launch/inference_launch.py
```

---

## 13. Важные команды

```bash
# Убить все процессы
pkill -9 -f gzserver; pkill -9 -f navigation_node; pkill -9 -f perception_node

# Сборка ROS
./build_ros.sh

# Сбор expert data
./run_drone.sh straight_cave.py
source learning/venv/bin/activate
python3 learning/collect_expert.py
```

---

## 14. Контакты и ссылки

- **GitHub:** Drone_offline_navigation
- **README:** `README.md`
- **Контекст проекта:** `PROJECT_CONTEXT.md`
- **Состояние обучения:** `learning/PROJECT_STATE.md`
