# Plan_actions.md — Статус реализации NN Control

## Общий прогресс

- [x] Этап 1: Структура `learning/` создана
- [x] Этап 2: `config.py`
- [x] Этап 3: `utils.py`
- [x] Этап 4: `reward.py`
- [x] Этап 5: `drone_env.py`
- [x] Этап 6: `collect_expert.py`
- [x] Этап 7: `bc_model.py`
- [x] Этап 8: `callbacks.py`
- [x] Этап 9: `train.py`
- [x] Этап 10: `inference_node.py` + `inference_launch.py`
- [x] Этап 11: `requirements.txt` + `.gitignore`
- [x] Этап 12: `training_launch.py` (для headless-тренировки)
- [x] Финальная проверка: все файлы, cross-references, синтаксис

---

## Детальный статус

| Файл | Статус | Примечание |
|------|--------|------------|
| `learning/config.py` | ✅ | Все параметры симуляции, BC, PPO, наград |
| `learning/utils.py` | ✅ | Управление Gazebo (kill/generate/launch/wait), сброс дрона |
| `learning/reward.py` | ✅ | Функция награды (7 компонентов) |
| `learning/drone_env.py` | ✅ | Gymnasium-среда (13-dim obs, 3-dim action, ROS 2 bridge) |
| `learning/collect_expert.py` | ✅ | ROS 2 нода для сбора экспертных данных |
| `learning/bc_model.py` | ✅ | BCNet MLP, обучение, загрузка весов в PPO |
| `learning/callbacks.py` | ✅ | ConsoleMonitor + CSVEpisodeLogger |
| `learning/train.py` | ✅ | BC → PPO пайплайн, --resume, --no-bc |
| `learning/inference_node.py` | ✅ | ROS 2 нода инференса, загрузка PPO, publish cmd_vel |
| `learning/launch/inference_launch.py` | ✅ | Копия simulation_launch.py, control_node → inference_node |
| `learning/launch/training_launch.py` | ✅ | Headless launch для тренировки (без control_node) |
| `learning/requirements.txt` | ✅ | torch, stable-baselines3, gymnasium, tensorboard |
| `learning/.gitignore` | ✅ | checkpoints/, tensorboard_logs/, expert_data/ |

---

## Проверка целостности

- ✅ Все 11 Python-файлов проходят `py_compile` (синтаксис без ошибок)
- ✅ Все импорты внутри `learning/` ссылаются на `config` и друг друга корректно
- ✅ Ни один существующий файл проекта (`src/`, `scripts/`, `run_drone.sh`, `PROJECT_CONTEXT.md`) не изменён
- ✅ Единственное изменение в git — `cave.world` (line endings) — не связано с нашей работой

## Команды для запуска

**Сбор экспертных данных:**
```bash
# Терминал 1: Запустить симуляцию в штатном режиме
./run_drone.sh

# Терминал 2: Запустить сбор данных
python3 learning/collect_expert.py
```

**Обучение (BC → PPO):**
```bash
cd learning && python3 train.py
```

**Продолжить обучение с чекпоинта:**
```bash
cd learning && python3 train.py --resume learning/checkpoints/rl_model_XXXXX_steps.zip
```

**Обучение без BC (PPO с нуля):**
```bash
cd learning && python3 train.py --no-bc
```

**Запуск инференса обученной модели:**
```bash
ros2 launch learning/launch/inference_launch.py
```

**Мониторинг:**
```bash
tensorboard --logdir learning/tensorboard_logs
```
