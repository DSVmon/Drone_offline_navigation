import random
import math
import os

def generate_cave(output_path, length=40.0, width=3.5, segment_length=1.5):
    """
    Генерирует SDF файл извилистой пещеры.
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

    # Текстура для всех элементов пещеры
    cave_material = "Gazebo/Bricks"
    
    # Генерируем 5 зон с экстремальными условиями
    extreme_zones = random.sample(range(5, num_segments - 10), 5)
    
    for i in range(num_segments):
        # 1. Горизонтальный изгиб
        current_angle += random.uniform(-0.35, 0.35)
        current_angle = max(-1.1, min(1.1, current_angle))
        
        # 2. Логика ширины и высоты (Экстремальные зоны)
        target_width = 4.0
        target_floor = 0.0
        target_ceiling = 3.5
        
        if i in extreme_zones:
            mode = random.choice(["narrow_1m", "wide_10m", "low_ceiling", "high_floor"])
            if mode == "narrow_1m":
                target_width = 1.0
            elif mode == "wide_10m":
                target_width = 10.0
            elif mode == "low_ceiling":
                target_ceiling = 0.8 # Опускаем потолок на ~70%
                target_floor = 0.0
            elif mode == "high_floor":
                target_floor = 1.5 # Ступенька на полу на 50%
                target_ceiling = 3.5
        else:
            # В обычных зонах небольшая вариативность
            target_width = current_width + random.uniform(-0.5, 0.5)
            target_width = max(2.5, min(6.0, target_width))
        
        # Резкие, но не мгновенные переходы для стабильности коллизий
        current_width = current_width * 0.5 + target_width * 0.5
        current_floor_z = current_floor_z * 0.5 + target_floor * 0.5
        current_ceiling_z = current_ceiling_z * 0.5 + target_ceiling * 0.5
        
        # Минимальный зазор безопасности для предотвращения схлопывания
        if current_ceiling_z - current_floor_z < 0.7:
            current_ceiling_z = current_floor_z + 0.7

        next_x = current_x + segment_length * math.cos(current_angle)
        next_y = current_y + segment_length * math.sin(current_angle)
        perp_angle = current_angle + math.pi / 2
        
        # --- ЦЕЛЬНЫЕ СТЕНЫ (с перекрытием для исключения разрывов) ---
        wall_height = 5.0 # Делаем стены высокими, чтобы дрон не вылетел сверху
        wall_overlap = 0.5 # Перекрытие сегментов для исключения щелей
        
        for side, side_name in [(1, "left"), (-1, "right")]:
            sx = next_x + (current_width/2) * side * math.cos(perp_angle)
            sy = next_y + (current_width/2) * side * math.sin(perp_angle)
            
            links += f"""
      <link name="{side_name}_wall_{i}">
        <pose>{sx} {sy} 1.5 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>{segment_length + wall_overlap} 0.4 {wall_height}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{segment_length + wall_overlap} 0.4 {wall_height}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{cave_material}</name></script></material>
        </visual>
      </link>"""

        # --- ПОТОЛОК И ПОЛ (тоже с перекрытием) ---
        for z_pos, name in [(current_ceiling_z, "ceiling"), (current_floor_z, "floor")]:
            links += f"""
      <link name="{name}_{i}">
        <pose>{next_x} {next_y} {z_pos} 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>{segment_length + wall_overlap} {current_width + 1.0} 0.2</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{segment_length + wall_overlap} {current_width + 1.0} 0.2</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{cave_material}</name></script></material>
        </visual>
      </link>"""

        # --- ГАРАНТИРОВАННЫЕ ПРЕПЯТСТВИЯ (СТОЛБЫ И КАМНИ) ---
        if i % 7 == 0:
            obs_x = next_x
            obs_y = next_y + random.uniform(-current_width/4, current_width/4)
            obs_h = random.uniform(1.0, 2.5)
            
            # Сталагмит или сталактит в зависимости от четности
            if i % 14 == 0: # Сталагмит (из пола)
                obs_z = current_floor_z + obs_h/2
            else: # Сталактит (с потолка)
                obs_z = current_ceiling_z - obs_h/2

            links += f"""
      <link name="obs_stone_{i}">
        <pose>{obs_x} {obs_y} {obs_z} 0 0 {random.uniform(0, 3.14)}</pose>
        <collision name="collision">
          <geometry><box><size>0.4 0.4 {obs_h}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.4 0.4 {obs_h}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Wood</name></script></material>
        </visual>
      </link>"""
        
        current_x, current_y = next_x, next_y
 
     # Тупик в конце (адаптирован под финальную ширину)
    links += f"""
      <link name="end_wall">
        <pose>{current_x + 0.2} {current_y} 1.5 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>0.2 {current_width + 0.4} 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.2 {current_width + 0.4} 3</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Bricks</name></script></material>
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
