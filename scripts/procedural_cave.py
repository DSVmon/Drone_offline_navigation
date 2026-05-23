import random
import math
import os

def generate_cave(output_path, length=100.0, width=4.5, segment_length=1.5):
    """
    Генерирует SDF файл сложной извилистой пещеры длиной 100 м.
    Включает: 8-12 резких перепадов высот (опуск потолка / подъём пола),
    переменную ширину (сужения до 2.0м и расширения до 8.0м),
    окно 0.8×0.8м в потолке, препятствия с зазором ≥80см до пола/потолка,
    крутые повороты, сталагмиты/сталактиты, горизонтальные плиты.
    """
    sdf_header = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="procedural_cave_world">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    
    <!-- Gazebo ROS State Plugin (World level) -->
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros>
        <namespace>/gazebo</namespace>
      </ros>
      <update_rate>10.0</update_rate>
    </plugin>

    <physics type="ode">
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
    </physics>
    <gui fullscreen='false'>
      <camera name='user_camera'>
        <pose>-6.0 0.0 4.0 0.0 0.5 0.0</pose>
        <view_controller>orbit</view_controller>
      </camera>
    </gui>
    <model name="procedural_cave">
      <static>true</static>
"""
    sdf_footer = """
    </model>
  </world>
</sdf>
"""
    current_x, current_y, current_angle = 0.0, 0.0, 0.0
    num_segments = int(length / segment_length)
    links = ""
    
    current_width = width
    current_floor_z = 0.0
    current_ceiling_z = 3.0
    
    # Direction bias — накапливается для плавных длинных поворотов
    turn_bias = 0.0
    
    # Текстура для всех элементов пещеры
    cave_material = "Gazebo/Bricks"
    
    # === 8-12 РЕЗКИХ ПЕРЕПАДОВ ВЫСОТ (опуск потолка / подъём пола / сжатие) ===
    sharp_count = random.randint(8, 12)
    sharp_indices = sorted(random.sample(range(6, num_segments - 8), sharp_count))
    
    # === ЗОНЫ КРУТЫХ ПОВОРОТОВ (дополнительные резкие изгибы) ===
    turn_count = random.randint(5, 7)
    remaining_for_turns = [i for i in range(4, num_segments - 6) if i not in sharp_indices]
    turn_indices = set(random.sample(remaining_for_turns, min(turn_count, len(remaining_for_turns))))
    
    # === ПЕРЕМЕННАЯ ШИРИНА: зоны сужения и расширения ===
    remaining = [i for i in range(5, num_segments - 6) if i not in sharp_indices and i not in turn_indices]
    random.shuffle(remaining)
    narrow_count = min(random.randint(4, 6), len(remaining) // 2)
    wide_count = min(random.randint(3, 5), (len(remaining) - narrow_count) // 2)
    narrow_indices = set(remaining[:narrow_count])
    wide_indices = set(remaining[narrow_count:narrow_count + wide_count])
    
    # Отверстие 0.8×0.8 м в потолке (окно наверх)
    hole_segment = random.randint(10, max(10, num_segments - 15))
    
    for i in range(num_segments):
        # 1. Горизонтальный изгиб (извилистый туннель)
        turn_bias += random.uniform(-0.2, 0.2)
        turn_bias = max(-0.55, min(0.55, turn_bias))
        if i in turn_indices:
            # Резкий поворот: дополнительный рывок угла
            current_angle += random.choice([-0.8, 0.8])
        else:
            current_angle += turn_bias * 0.35 + random.uniform(-0.18, 0.18)
        current_angle = max(-1.1, min(1.1, current_angle))
        
        # 2. Логика ширины и высоты
        target_width = current_width
        target_floor = 0.0
        target_ceiling = 3.5
        use_smooth = 0.4
        
        if i in sharp_indices:
            # Резкий перепад: потолок вниз, пол вверх, или оба сразу
            mode = random.choice(["ceiling", "floor", "both"])
            if mode == "ceiling":
                prev = current_ceiling_z
                target_ceiling = prev - random.uniform(0.8, 2.2)
                target_ceiling = max(current_floor_z + 1.0, target_ceiling)
            elif mode == "floor":
                target_floor = current_floor_z + random.uniform(0.6, 1.8)
                target_floor = min(current_ceiling_z - 1.0, target_floor)
            else:  # both — сжатие сверху и снизу одновременно
                target_ceiling = current_ceiling_z - random.uniform(0.6, 1.5)
                target_floor = current_floor_z + random.uniform(0.4, 1.2)
                target_ceiling = max(target_floor + 1.0, target_ceiling)
                target_floor = min(target_ceiling - 1.0, target_floor)
            target_width = current_width + random.uniform(-0.3, 0.3)
            use_smooth = 0.9  # очень резкий переход
        elif i in narrow_indices:
            target_width = random.uniform(2.0, 2.8)
            target_ceiling = current_ceiling_z
        elif i in wide_indices:
            target_width = random.uniform(6.0, 8.0)
            target_ceiling = current_ceiling_z
        else:
            target_width = current_width + random.uniform(-0.4, 0.4)
            target_width = max(3.5, min(5.5, target_width))
        
        current_width = current_width * (1 - use_smooth) + target_width * use_smooth
        current_floor_z = current_floor_z * (1 - use_smooth) + target_floor * use_smooth
        current_ceiling_z = current_ceiling_z * (1 - use_smooth) + target_ceiling * use_smooth
        
        # Минимальный зазор безопасности
        if current_ceiling_z - current_floor_z < 0.8:
            current_ceiling_z = current_floor_z + 0.8
        # Максимальная высота
        if current_ceiling_z > 5.0:
            current_ceiling_z = 5.0
        if current_floor_z < 0.0:
            current_floor_z = 0.0

        next_x = current_x + segment_length * math.cos(current_angle)
        next_y = current_y + segment_length * math.sin(current_angle)
        
        # --- СТЕНЫ ---
        wall_height = 5.0
        wall_overlap = 1.5
        wall_thickness = 0.8
        
        wall_length = segment_length + wall_overlap
        
        for side, side_name in [(1, "left"), (-1, "right")]:
            sx = next_x - side * (current_width/2) * math.sin(current_angle)
            sy = next_y + side * (current_width/2) * math.cos(current_angle)
            
            links += f"""
      <link name="{side_name}_wall_{i}">
        <pose>{sx} {sy} 1.5 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>{wall_length} {wall_thickness} {wall_height}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{wall_length} {wall_thickness} {wall_height}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{cave_material}</name></script></material>
        </visual>
      </link>"""

        # --- ЗАПОЛНИТЕЛЬ ВНЕШНИХ СТЫКОВ ---
        if i > 0:
            outer_offset = current_width/2 + wall_thickness
            for side, side_name in [(1, "left"), (-1, "right")]:
                fx = next_x - side * outer_offset * math.sin(current_angle)
                fy = next_y + side * outer_offset * math.cos(current_angle)
                links += f"""
      <link name="fill_{side_name}_{i}">
        <pose>{fx} {fy} 1.5 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>{wall_length} {wall_thickness} {wall_height}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{wall_length} {wall_thickness} {wall_height}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{cave_material}</name></script></material>
        </visual>
      </link>"""

        # --- ПОТОЛОК И ПОЛ — с усиленным перекрытием для компенсации поворотов ---
        ceiling_floor_length = segment_length + wall_overlap + 1.5  # +1.5м для покрытия щелей на поворотах
        ceiling_z_pos = current_ceiling_z - 0.1  # плита потолка чуть ниже
        floor_z_pos = current_floor_z + 0.1      # плита пола чуть выше
        
        # Потолок — сплошной, кроме сегмента с окном 0.8×0.8м
        if i == hole_segment:
            hole_gap = 0.8
            tile_total_w = current_width + 1.0  # общая ширина плиты потолка
            tile_total_l = ceiling_floor_length
            # 4 угловых плитки вокруг окна 0.8×0.8м
            tw = (tile_total_w - hole_gap) / 2  # ширина одной плитки (вдоль перпендикуляра)
            tl = (tile_total_l - hole_gap) / 2  # длина одной плитки (вдоль туннеля)
            perp_x = math.cos(current_angle)
            perp_y = math.sin(current_angle)
            for side_lr, sw in [(-1, tw), (1, tw)]:
                for side_fb, sl in [(-1, tl), (1, tl)]:
                    ox = -side_lr * (hole_gap/2 + sw/2) * perp_x
                    oy = -side_lr * (hole_gap/2 + sw/2) * perp_y
                    fx = side_fb * (hole_gap/2 + sl/2) * math.cos(current_angle)
                    fy = side_fb * (hole_gap/2 + sl/2) * math.sin(current_angle)
                    links += f"""
      <link name="ceiling_hole_{i}_{side_lr}_{side_fb}">
        <pose>{next_x + ox + fx} {next_y + oy + fy} {ceiling_z_pos} 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>{sl} {sw} 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sl} {sw} 0.2</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{cave_material}</name></script></material>
        </visual>
      </link>"""
        else:
            links += f"""
      <link name="ceiling_{i}">
        <pose>{next_x} {next_y} {ceiling_z_pos} 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>{ceiling_floor_length} {current_width + 1.0} 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{ceiling_floor_length} {current_width + 1.0} 0.2</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{cave_material}</name></script></material>
        </visual>
      </link>"""
        
        # Пол
        links += f"""
      <link name="floor_{i}">
        <pose>{next_x} {next_y} {floor_z_pos} 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>{ceiling_floor_length} {current_width + 1.0} 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{ceiling_floor_length} {current_width + 1.0} 0.2</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{cave_material}</name></script></material>
        </visual>
      </link>"""

        # --- ПРЕПЯТСТВИЯ (СТОЛБЫ, ПЛИТЫ, КАМНИ) ---
        if i >= 1:
            # Сталагмиты и сталактиты (каждый 7-й)
            if i % 7 == 0:
                obs_x = next_x
                obs_y = next_y + random.uniform(-current_width/3, current_width/3)
                clearance = current_ceiling_z - current_floor_z
                # Минимум 80см от края столба до потолка/пола
                max_obs_h = max(0.3, min(2.5, clearance - 0.8))
                obs_h = random.uniform(0.5, max_obs_h)
                
                if i % 8 == 0:  # Сталагмит (из пола)
                    obs_z = current_floor_z + obs_h/2
                else:  # Сталактит (с потолка)
                    obs_z = current_ceiling_z - obs_h/2

                links += f"""
      <link name="obs_stone_{i}">
        <pose>{obs_x} {obs_y} {obs_z} 0 0 {random.uniform(0, 3.14)}</pose>
        <collision name="collision">
          <geometry><box><size>{random.uniform(0.25, 0.45)} {random.uniform(0.25, 0.45)} {obs_h}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{random.uniform(0.25, 0.45)} {random.uniform(0.25, 0.45)} {obs_h}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Wood</name></script></material>
        </visual>
      </link>"""
            
            # Горизонтальные препятствия (плиты) — редко
            if i % 15 == 0 and i >= 3:
                slab_side = random.choice([-1, 1])
                slab_x = next_x - slab_side * (current_width * 0.25) * math.sin(current_angle)
                slab_y = next_y + slab_side * (current_width * 0.25) * math.cos(current_angle)
                slab_z = random.uniform(current_floor_z + 0.5, current_ceiling_z - 0.5)
                slab_w = random.uniform(0.8, 1.8)
                slab_h = random.uniform(0.1, 0.3)
                
                links += f"""
      <link name="obs_slab_{i}">
        <pose>{slab_x} {slab_y} {slab_z} 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>0.3 {slab_w} {slab_h}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.3 {slab_w} {slab_h}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Wood</name></script></material>
        </visual>
      </link>"""
        
        current_x, current_y = next_x, next_y
 
    # Тупик в конце (адаптирован под финальную ширину и высоту)
    dead_end_height = max(0.7, current_ceiling_z - current_floor_z)
    dead_end_z = (current_ceiling_z + current_floor_z) / 2
    links += f"""
      <link name="end_wall">
        <pose>{current_x + 0.2} {current_y} {dead_end_z} 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>0.2 {current_width + 0.4} {dead_end_height}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.2 {current_width + 0.4} {dead_end_height}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{cave_material}</name></script></material>
        </visual>
      </link>"""

    # Убеждаемся, что директория существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(sdf_header + links + sdf_footer)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_cave(sys.argv[1])
    else:
        print("Usage: python3 procedural_cave.py <output_world_path>")
