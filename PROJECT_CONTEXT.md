# Drone Offline Navigation: Phase 4.2 (Procedural Caves & Physical Turns)

Этот файл содержит полное описание текущего состояния проекта автономного спасательного дрона для обеспечения преемственности в новых сессиях разработки.

## Цель проекта
Разработка автономной системы навигации дрона в закрытых пространствах (пещерах) с использованием стереозрения (OpenCV) в качестве основного сенсора. dToF лазерный дальномер отключён (закомментирован).

## Технический стек
- **ОС:** ROS 2 Humble
- **Симулятор:** Gazebo Classic
- **Языки:** C++ (навигация/перцепция), Python (управление)
- **Библиотеки:** OpenCV 4.x, `cv_bridge`, `image_geometry`, `message_filters`
- **Среда:** Прямой запуск в Ubuntu 22.04 (WSL2) с поддержкой аппаратного ускорения (vGPU). Docker удалён для повышения производительности графики.

## Архитектура системы (Ноды)
1. **`navigation_node` (C++):** Обрабатывает стереопару, вычисляет карту диспаратности и определяет дистанцию в 5 зонах (Left, Center, Right, Top, Bottom). Публикует данные в `/navigation_node/stereo_distances`.
2. **`perception_node` (C++):** Обрабатывает данные с лазерных дальномеров dToF. **dToF отключён** на уровне подписки (закомментирован в `control_node`).
3. **`control_node` (Python):** Логика поведения. Конечный автомат: SEARCHING → INSPECTING → TURNING. **Гибридное 3D-управление:** X, Y и Yaw через `planar_move`, Z через сервис Gazebo.

## Ключевые технические решения

### 1. Процедурная генерация мира
- **`scripts/procedural_cave.py`:** Генерирует случайную извилистую пещеру при каждом запуске.
- **Параметры генерации:** длина 100м, сегмент 1.5м, стартовая ширина 4.5м.
- **Особенности (гарантированы в каждой генерации):**
  - **8–12 резких перепадов высот** — ceiling drop (0.8–2.2м), floor rise (0.6–1.8м), или both (сжатие с двух сторон). `smooth = 0.9` (очень резкий).
  - **5–7 зон крутых поворотов** — дополнительный рывок угла ±0.8 рад за сегмент. Основной изгиб: bias ±0.55, шаг bias×0.35 + rand±0.18, угол ограничен ±1.1 рад.
  - **4–6 узких зон** (ширина 2.0–2.8м) и **3–5 широких зон** (6.0–8.0м).
  - **Окно 0.8×0.8м** в потолке (4 угловые плитки вокруг отверстия, а не 2 боковые).
  - **Препятствия:** сталактиты/сталагмиты (каждый 7-й сегмент). Высота ограничена `clearance − 0.8` (минимум 80см зазор до пола/потолка). Плиты (каждый 15-й сегмент, со смещением от центра).
- **Перекрытие плит:** `ceiling_floor_length = segment_length + wall_overlap + 1.5` (4.5м), чтобы исключить щели на поворотах до 63°.
- **Двухслойные стены:** Каждый сегмент содержит основную стену (offset `width/2`) и внешний заполнитель (offset `width/2 + 0.8`), который добавляется на стыках сегментов (i > 0). Два слоя (по 0.8м) образуют стену толщиной 1.6м. Щели во внутреннем и внешнем слоях не совпадают — дрон не может пролететь сквозь стык на изгибе. Заполнители находятся снаружи стен и не сужают проход.
- **`wall_overlap = 1.5`, `wall_thickness = 0.8`, `wall_height = 5.0`**.
- **Чистый проход:** `номинальная_ширина − 0.8` (из-за толщины стен). Минимальный чистый проход 1.2м (узкие зоны 2.0–2.8м → 1.2–2.0м), что даёт запас 0.2–0.6м с каждой стороны.
- **Первое препятствие:** `i >= 2` (не ближе 3.0м от старта), чтобы дрон успел построить карту глубины.
- **Камера симуляции:** позиция `(-6, 0, 4)` с orbit-контроллером.

### 2. Гибридная 3D физика (Z-axis Workaround)
- **Проблема:** Стандартные плагины ROS 2 Humble (типа `planar_move`) не поддерживают ось Z, а `force_based_move` отсутствует.
- **Решение:** `libgazebo_ros_planar_move.so` для X/Y/Yaw, сервис `/gazebo/set_entity_state` для Z.
- **Z-сервис:** Заморозка X/Y на позиции входа во время INSPECTING (фазы ≥ 2), чтобы Gazebo не дёргал дрон горизонтально.

### 3. Миссия "Челночное движение" (Shuttle Mode)
- **SEARCHING:** Единственное движущееся состояние (полёт вперёд/назад). Использует `_calculate_smooth_3d_speed` для реактивного уклонения от стен (`l_rep/r_rep` для рыскания, `t_rep/b_rep` для высоты).
- **INSPECTING с многопозиционным зондированием:** Остановка при `center < 0.8м`. Два полных прохода:
  - **Pass 0:** от позиции входа. Фазы: INIT → PROBE_UP → SWEEP_L_HI → RET_YAW → SWEEP_R_HI → RET_YAW → GO_MID → SWEEP_L_MID → RET_YAW → SWEEP_R_MID → RET_YAW → PROBE_DOWN → SWEEP_L_LO → RET_YAW → SWEEP_R_LO → RET_YAW → RET_Z → BACKUP.
  - **Pass 1 (backup):** отступ назад (1.5с), sweep_angle × 1.45, повтор фаз PROBE_UP...RET_Z → DECIDE.
  - **PROBE_UP:** `entry_z + entry_top × 0.7` (70% до потолка, clamped 0.3–3.2).
  - **PROBE_DOWN:** `entry_z − entry_bottom × 0.7` (70% до пола, clamped 0.5–3.2).
  - **Адаптивность без фиксированных отступов:** все вертикальные цели считаются от реальных показаний стереокамеры.
  - **Сброс Z-ramp при смене фазы:** `target_z = current_z`, обнуление `vz/vx/yaw_cmd` для предотвращения ухода через потолок/пол.
- **DECIDE (фаза 19):**
  - **Уровень 1:** передние снимки (`baseline`, `high`, `low`, `backup`, `*_min`, их `b_*` варианты). Если проходим → SEARCHING (без смены курса).
  - **Уровень 2:** yaw-снимки (`high/mid/low_left/right_yaw`). Если проходим → TURNING в ту сторону, затем SEARCHING.
  - **Критерии проходимости:** `center ≥ 0.3` И `min(left,right) ≥ 0.4` И `min(top,bottom) ≥ 0.25`. Все три обязательны.
  - **Коррекция высоты для `high`/`low` снимков:** `height_ok` считается по просвету на **входной высоте** (`entry_top`/`entry_bottom`), а не на зондовой — это позволяет находить путь над/под препятствием, даже когда зонд слишком близко к потолку/полу.
  - **Score:** `0.5×center + 0.3×min(l,r) + 0.2×min(t,b)`.
  - Если ни один снимок не прошёл → **DEAD END** → TURNING (180°).
- **Жёсткий таймаут:** 55с на весь INSPECTING → принудительный DEAD END.
- **Stuck detection:** если INSPECTING → SEARCHING → снова INSPECTING на том же месте (< 0.2м), счётчик растёт. При ≥ 2 → принудительный DEAD END без повторного INSPECTING.
- **Пропуск фаз при коллизии в INSPECTING:** Столкновение → переход на следующую безопасную фазу (≤6 → GO_MID, ≤11 → PROBE_DOWN, иначе → BACKUP). Ранее собранные `inspect_data` сохраняются.
- **TURNING:** Разворот на target_yaw. При завершении (допуск 0.1 рад) → SEARCHING. При `returned_home=True` увеличивает lap_count.
- **Virtual Wall:** `signed_along_cave = dot(position, entrance_heading) < -0.3` → TURNING. Guard `wall_hit_guard` предотвращает повтор.
- **Proximity Finish:** `dist_to_home < 1.5` или `signed_along_cave < 0.5` после 3с в SEARCHING → TURNING. Guard по `last_searching_entry_time`.

### 4. Стабильность и плавность
- **EMA-фильтрация (alpha=0.3)** для стереоданных.
- **Минимальная скорость:** 0.25 м/с.
- **Гистерезис INSPECTING:** вход < 0.8м, выход через DECIDE.
- **Сглаживание команд (ramping):** alpha = 0.3 для VX, VZ, Yaw.
- **STATUS** раз в 5 секунд отдельным таймером.

### 5. Обнаружение столкновений
- **Gazebo contact sensor:** Основной — активен всегда.
- **Velocity discrepancy:** `cmd_speed − odom_speed > 0.4` для ≥ 6 кадров → WALL.
- **Z-jam:** `|vz_cmd| > 0.3` но `|dz| < 0.005` для ≥ 10 кадров → CEILING/FLOOR.
- **Sustained proximity:** `min(all 5) < 0.15` для ≥ 6 кадров → OBSTACLE.
- **Orientation-based:** roll/pitch rate > 1.2 rad/s → тип по `vz_cmd`.
- **Все продвинутые детекторы отключены в INSPECTING** (ложные срабатывания от штатных движений зондирования).
- **Collision type tracking:** NONE / WALL / CEILING / FLOOR / OBSTACLE / CONTACT.

### 6. Логирование
- **CSV:** `logs/flight_last.csv`. Колонки: timestamp, state, x, y, z, roll/pitch/yaw_deg, dist_left/center/right/top/bottom, dtof_alert, collision_type, best_dir, max_space, cmd_vx/vz/yaw, lap, total_hits.
- **Консоль:** `[WALL]`, `[CEILING]`, `[CONTACT]`, `[TIGHT]`, `[HIT #N]`, `[DEAD END]`, `[TURN]`, `[STATUS]`.

## Решённые проблемы (Lessons Learned)
- **Stereo Blindness:** При подлёте < 0.3м стереокамера выдаёт ложные 10м. EMA-фильтрация смягчает.
- **Frozen Z:** `planar_move` не поддерживает Z. Исправлено гибридным методом (сервис Gazebo).
- **Infinite Wall Spin:** `wall_hit_guard`.
- **Wall Gaps:** Двухслойные стены + увеличенный `ceiling_overlap` для потолка/пола.
- **Proximity Finish Loop:** Guard `last_searching_entry_time` (3с).
- **Collision False Positives in INSPECTING:** Детекторы отключены в INSPECTING.
- **Vertical Probe Ceiling Loop:** При коллизии — пропуск фазы, а не сброс в 1.
- **Dead-end Loop (INSPECT → SEARCH → INSPECT):** Двухуровневый DECIDE + yaw-разворот через TURNING.
- **Slab-in-Middle False DEAD END:** `height_ok` для probe-снимков считает по `entry_top`/`entry_bottom`, а не по probe-позиции.
- **Узкие проходы после учёта толщины стен:** Минимальная ширина зон поднята с 1.2м до 2.0м (чистый проход 1.2м).
- **Щели в потолке на поворотах:** +1.5м к длине плит перекрытия.

## Команды для запуска
```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch drone_simulation simulation_launch.py
```

## Текущий статус
Система верифицирована. Дрон циклически летает по процедурно генерируемой пещере (100м, 66 сегментов). Маршрут: SEARCHING → [INSPECTING при препятствии] → SEARCHING (найден проход) или TURNING (тупик/разворот на проход). DECIDE имеет двухуровневый приоритет: сначала передние снимки (прямой полёт), затем yaw-снимки (разворот в проход). Проект готов к внедрению SLAM (Phase 5).
