# 🛡️ План улучшения: Умное расширение режима INSPECTING (Phase 4.3)

## 📌 Краткое резюме

Расширение существующего 20-фазного INSPECTING режима для более активного физического исследования всех 3D направлений движения дрона перед принятием решения о маршруте.

**Текущее:** Дрон вращается на месте (yaw sweeps) и анализирует снимки
**Новое:** Дрон физически смещается влево/вправо и в диагональные углы для комплексного анализа

---

## 🎯 Три уровня исследования

### 1️⃣ PRIMARY SCAN (фазы 2-7) - Уже реализовано, без изменений
- Поднятие на 70% к потолку (top зона)
- Возврат в центр
- Снижение на 70% к полу (bottom зона)
- Возврат в центр

**Результат:** Оценка forward проходимости на трех высотах (high, mid, low)

---

### 2️⃣ LATERAL SWEEP (фазы 9-13) - **НОВОЕ**
- **PHASE 9:** Проверка: если center < 0.5м → переходить к боковому исследованию
- **PHASE 10:** Сдвиг влево Y-0.3м, фиксирование distances
- **PHASE 11:** Возврат в центр
- **PHASE 12:** Сдвиг вправо Y+0.3м, фиксирование distances
- **PHASE 13:** Возврат в центр

**Результат:** Определение наличия боковых проходов, когда центр заблокирован

---

### 3️⃣ DIAGONAL PROBING (фазы 14-17) - **НОВОЕ**
- **PHASE 14:** Движение up+left (Z+0.5м, Y-0.3м)
- **PHASE 15:** Возврат в baseline
- **PHASE 16:** Движение up+right (Z+0.5м, Y+0.3м)
- **PHASE 17:** Возврат в baseline

**Результат:** Проверка верхних углов для узких мест с низким потолком

---

## 💾 Расширенная структура inspect_data

```python
inspect_data = {
    # === СУЩЕСТВУЮЩИЕ (фазы 1-7) ===
    "baseline": distances,           # Начальный снимок
    "high": distances,               # После подъема
    "low": distances,                # После снижения
    "high_left_yaw": distances,      # Разворот влево у потолка
    "high_right_yaw": distances,     # Разворот вправо у потолка
    "mid_left_yaw": distances,       # Разворот влево в центре
    "mid_right_yaw": distances,      # Разворот вправо в центре
    "low_left_yaw": distances,       # Разворот влево у пола
    "low_right_yaw": distances,      # Разворот вправо у пола
    "backup": distances,             # Перед pass 1
    
    # === НОВЫЕ (фазы 9-17) ===
    "left_probe": distances,         # Y-0.3м боковой проход
    "right_probe": distances,        # Y+0.3м боковой проход
    "diagonal_up_left": distances,   # Z+0.5м, Y-0.3м
    "diagonal_up_right": distances,  # Z+0.5м, Y+0.3м
}
```

---

## 🤖 Переработанный алгоритм анализа (analyze_all_paths)

### Выход: Список всех найденных путей

```python
analysis_result = {
    "paths": [
        {
            "direction": "forward_baseline",      # или "forward_up", "forward_down"
            "center_dist": 1.2,
            "side_dist": 0.8,
            "vertical_dist": 0.5,
            "safety_score": 0.75,
            "confidence": 0.85,
            "passable": True
        },
        {
            "direction": "forward_left",         # Y-0.3м смещение
            "center_dist": 0.7,
            "safety_score": 0.65,
            "confidence": 0.70,
            "passable": True
        },
        # ... еще пути ...
    ],
    "best_path": "forward_baseline",            # Рекомендуемый
    "confidence": 0.85,                         # Уверенность в решении
    "deadlock": False,                          # Есть ли пути?
    "recommendation": "forward"                 # Действие: forward/turn_left/retreat
}
```

### Приоритет выбора пути

1. **Лучший forward путь** (baseline, high, low, левый, правый)
2. **Диагональный путь** (up-left, up-right), если forward заблокирован
3. **Yaw turn** (как было), если forward и diagonal заблокированы
4. **Retreat** (TURNING), если ничего не проходит

---

## ⏱️ Общая временная шкала

```
PHASE 1:           INIT (baseline snapshot)              ~ 0.5 сек
PHASE 2-7:         PRIMARY SCAN (up/down)                ~ 6 сек
PHASE 9-13:        LATERAL SWEEP (left/right)           ~ 4 сек
PHASE 14-17:       DIAGONAL PROBING (up-left/up-right)  ~ 4 сек
PHASE 18:          ANALYZE ALL PATHS                    ~ 1 сек
PHASE 19:          DECIDE & TRANSITION                  ~ 0.5 сек
─────────────────────────────────────────────────────────────
ВСЕГО:             ~16 сек (оптимизируется параллельными движениями)
```

---

## 📊 Адаптивные пороги (Adaptive Thresholds)

### Сценарий: Узкий по ширине (narrow_horizontal)
```
center_required: 0.4м
width_required: 0.3м (min left/right)
min_paths: 1
→ Если есть хотя бы один path с этими параметрами → forward
```

### Сценарий: Узкий по высоте (narrow_vertical)
```
center_required: 0.4м
height_required: 0.3м (min top/bottom)
min_paths: 1
→ Ищем paths вверх или вниз
```

### Сценарий: Тупиковая стена (dead_end)
```
center_required: 0.5м заглубление
min_paths: 2+
confidence_required: 0.8
→ Если нет ≥2 путей с высокой уверенностью → RETREAT
```

### Сценарий: Много препятствий (multi_obstacle)
```
center_required: 0.5м
width_required: 0.3м
height_required: 0.3м
min_paths: 2+
confidence_required: 0.7
→ Требуется наибольшая уверенность перед выбором
```

---

## 🔧 Технические детали интеграции

### Что НЕ меняется
- FSM структура (SEARCHING → INSPECTING → TURNING)
- Двухпроходная система (pass0 с 90° углами, pass1 с 145°)
- EMA фильтрация стереосенсоров
- Gazebo Z-control
- Collision detection

### Что добавляется
- **Фазы 9-17:** Новая логика боковых и диагональных движений
- **analyze_all_paths():** Полная переработка алгоритма анализа
- **scenario_detection():** Определение типа препятствия (adaptive thresholds)
- **CSV logging:** Новые колонки - path_count, best_direction, confidence, scenario

### Минимальные изменения
- inspect_probe_phase логика расширена (добавить фазы 9-17)
- inspect_data структура расширена (новые snapshot ключи)
- _calculate_smooth_3d_speed() не меняется

---

## ✅ Критерии успеха

- ✅ Дрон различает **узкий проход** от **тупиковой стены**
- ✅ Боковые пути активно исследуются, когда center < 0.5м
- ✅ Диагональные углы проверяются для низких потолков
- ✅ Confidence метрика > 0.7 перед выбором маршрута
- ✅ Dead-end detection надежен (false positives < 5%)
- ✅ Время анализа < 20 сек
- ✅ Успешный проход через procedural cave с препятствиями > 80%

---

## 🚀 Фазы реализации

**Фаза 1 (базовое):** Lateral sweep (фазы 9-13)
- Добавить 4 новые фазы Y-смещения
- Minimal code changes

**Фаза 2 (улучшенное):** Diagonal probing + новый scoring
- Добавить 4 диагональные фазы
- Полная переработка analyze_all_paths()

**Фаза 3 (продвинутое):** Adaptive thresholds + scenario detection
- Определение типа препятствия
- Контекстные пороги

---

## 📁 Файлы для модификации

1. **src/drone_control/drone_control/control_node.py**
   - Расширить inspect_probe_phase диапазон
   - Добавить фазы 9-17 логику
   - Переработать analyze_all_paths() (фаза 19)

2. **src/drone_navigation/src/navigation_node.cpp**
   - Без изменений (стереозрение работает как раньше)

3. **scripts/procedural_cave.py**
   - Без изменений (генератор пещер не меняется)

---

## 📚 Дополнительные ресурсы

- **Полный план:** IMPROVED_INSPECTING_PLAN.md
- **Анализ проекта:** PROJECT_ANALYSIS.md
- **История разработки:** PROJECT_CONTEXT.md
