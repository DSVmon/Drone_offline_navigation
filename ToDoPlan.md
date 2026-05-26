# План запуска обучения (WSL2 + zsh)

## Шаг 0. Настройка ROS в zsh

```bash
source /opt/ros/humble/setup.zsh
```

Если хочешь, чтобы ROS подхватывался автоматически в новом терминале — добавь эту строку в `~/.zshrc`:
```bash
echo "source /opt/ros/humble/setup.zsh" >> ~/.zshrc
```

## Шаг 1. Установка зависимостей

```bash
# Активируй ROS (если не прописал в .zshrc)
source /opt/ros/humble/setup.zsh

# Создание виртуального окружения
sudo apt install python3-venv -y
python3 -m venv learning/venv

# Активация venv
source learning/venv/bin/activate

# Установка библиотек
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3 gymnasium tensorboard numpy
```

## Шаг 2. Проверка сборки

```bash
source /opt/ros/humble/setup.zsh
colcon build --symlink-install
source install/setup.zsh
./run_drone.sh        # Ctrl+C после проверки
```

## Шаг 3. Сбор экспертных данных (2 терминала)

**Терминал A:**
```bash
source /opt/ros/humble/setup.zsh
source install/setup.zsh
./run_drone.sh
```

**Терминал B:**
```bash
source learning/venv/bin/activate
python3 learning/collect_expert.py
```

Ждать ~10 мин до `[EXPERT] Target reached`.

## Шаг 4. Запуск обучения

```bash
source learning/venv/bin/activate
cd learning && python3 train.py
```

~5-8 часов.

## Шаг 5. Мониторинг (терминал C)

```bash
source learning/venv/bin/activate
tensorboard --logdir learning/tensorboard_logs
```

Браузер: `http://localhost:6006`

## Шаг 6. Инференс

```bash
source /opt/ros/humble/setup.zsh
source install/setup.zsh
ros2 launch learning/launch/inference_launch.py
```
