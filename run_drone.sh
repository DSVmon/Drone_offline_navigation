#!/bin/bash

# Скрипт автоматической сборки и запуска Drone Offline Navigation
# Использование: ./run_drone.sh

# 1. Проверка окружения ROS 2
if [ -z "$ROS_DISTRO" ]; then
    echo "Ошибка: ROS 2 не обнаружен. Пожалуйста, выполните 'source /opt/ros/humble/setup.bash' сначала."
    exit 1
fi

echo "--- Начинаем сборку проекта ---"

# 2. Сборка проекта
# Используем --symlink-install для Python нод (изменения применяются без пересборки)
colcon build --symlink-install

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

# 4. Запуск симуляции
echo "--- Запуск симуляции (Mission Phase 4.2) ---"
ros2 launch drone_simulation simulation_launch.py
