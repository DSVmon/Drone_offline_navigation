# Drone Offline Navigation: Phase 4.2 (Procedural Caves)

Данный репозиторий содержит систему автономной навигации для спасательного дрона, предназначенного для исследования закрытых пространств (шахт, пещер) без GPS.

## Ключевые возможности (Phase 4.2)
- **Stereo Navigation:** 5-зонный анализ глубины в реальном времени (OpenCV + StereoBM).
- **Procedural Environments:** Генерация уникальных извилистых пещер с препятствиями (сталактитами) при каждом запуске.
- **3D Reactive Control:** Реактивное уклонение от стен и препятствий по трем осям (X, Z, Yaw).
- **Mission Logic:** Автономный цикл "Поиск -> Осмотр -> Разворот -> Возврат".
- **Flight Logging:** Сбор телеметрии и данных сенсоров в CSV для последующего анализа.
- **Native ROS 2:** Работа напрямую в ОС (Ubuntu 22.04 / WSL2) для максимальной производительности и поддержки GPU.

## Архитектура системы
- **`drone_navigation` (C++):** Обработка стереопары, компенсация наклона (IMU), расчет дистанций.
- **`drone_control` (Python):** Конечный автомат миссии, P-контроллер разворота, 3D-руление.
- **`drone_perception` (C++):** Резервный мониторинг через dToF дальномеры.
- **`drone_simulation`:** Модель дрона (URDF) и миры Gazebo.

## Быстрый запуск (Quick Launch)

### Подготовка (Windows + WSL2)
Проект оптимизирован для работы в **WSL2 (Ubuntu 22.04)** с установленным **ROS 2 Humble**.

### Запуск симуляции
1. Соберите проект:
```bash
colcon build --symlink-install
source install/setup.bash
```

2. Запустите миссию:
```bash
ros2 launch drone_simulation simulation_launch.py
```

После запуска:
1. Откроется окно **Gazebo** с процедурно сгенерированной пещерой.
2. Откроются окна **OpenCV** с визуализацией rectified-изображения и карты диспаратности.
3. Дрон начнет миссию: взлет, полет по пещере, разворот в тупике и возврат к старту.

## Структура логов
Данные полета сохраняются в папку `logs/` в формате CSV. Расширенный лог включает:
`timestamp,state,x,y,z,roll,pitch,yaw,dist_left,dist_center,dist_right,dist_top,dist_bottom,dtof_alert,collision_event,cmd_vx,cmd_vz,cmd_yaw`

## Разработка
Для внесения изменений в логику:
- **C++ (Навигация):** `src/drone_navigation/src/navigation_node.cpp`
- **Python (Управление):** `src/drone_control/drone_control/control_node.py`
- **Генерация мира:** `scripts/procedural_cave.py`
