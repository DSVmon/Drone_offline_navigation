#!/bin/bash

# Скрипт автоматической сборки и запуска Drone Offline Navigation
# Использование: ./run_drone.sh [cave_script]
# Примеры:
#   ./run_drone.sh                      # процедурная пещера (по умолчанию)
#   ./run_drone.sh straight_cave.py     # прямой туннель (для expert data)
#   ./run_drone.sh gentle_cave.py       # плавные повороты

CAVE_SCRIPT=${1:-procedural_cave.py}

# 1. Проверка окружения ROS 2
if [ -z "$ROS_DISTRO" ]; then
    echo "Ошибка: ROS 2 не обнаружен. Пожалуйста, выполните 'source /opt/ros/humble/setup.bash' сначала."
    exit 1
fi

echo "--- Начинаем сборку проекта ---"

# 2. Сборка проекта
COLCON_PYTHON_SETUP_PY_EXTENSION_DISABLED=1 colcon build --symlink-install \
  --packages-select drone_control drone_navigation drone_perception drone_simulation

# Проверка успешности сборки
if [ $? -ne 0 ]; then
    echo "Ошибка: Сборка провалилась. Проверьте сообщения выше."
    exit 1
fi

echo "--- Сборка завершена успешно ---"

# 3. Активация окружения воркспейса
if [ -f "install/setup.bash" ]; then
    source install/setup.bash
    echo "--- Окружение воркспейса активировано ---"
else
    echo "Ошибка: Файл install/setup.bash не найден!"
    exit 1
fi

# 4. Запуск симуляции с выбранной пещерой
echo "--- Запуск симуляции (cave: $CAVE_SCRIPT) ---"
ros2 launch drone_simulation simulation_launch.py cave_script:=$CAVE_SCRIPT
