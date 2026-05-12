import random
import math
import os

def generate_cave(output_path, length=40.0, width=3.5, segment_length=1.5):
    """
    Генерирует SDF файл извилистой пещеры с добавлением препятствий.
    """
    sdf_header = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="procedural_cave_world">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
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
    max_turn = 0.25 # радианы
    num_segments = int(length / segment_length)
    links = ""
    
    # Константы высоты для базовой версии
    CEILING_Z = 3.0
    FLOOR_Z = 0.0
    
    for i in range(num_segments):
        # Случайный изгиб
        current_angle += random.uniform(-max_turn, max_turn)
        current_angle = max(-0.7, min(0.7, current_angle)) # не даем развернуться назад
        
        next_x = current_x + segment_length * math.cos(current_angle)
        next_y = current_y + segment_length * math.sin(current_angle)
        
        perp_angle = current_angle + math.pi / 2
        # Левая и правая стены
        for side, side_name in [(1, "left"), (-1, "right")]:
            sx = next_x + (width/2) * side * math.cos(perp_angle)
            sy = next_y + (width/2) * side * math.sin(perp_angle)
            
            links += f"""
      <link name="{side_name}_wall_{i}">
        <pose>{sx} {sy} 1.5 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>{segment_length + 0.1} 0.2 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{segment_length + 0.1} 0.2 3</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Bricks</name></script></material>
        </visual>
      </link>"""

        # Потолок
        links += f"""
      <link name="ceiling_{i}">
        <pose>{next_x} {next_y} {CEILING_Z} 0 0 {current_angle}</pose>
        <visual name="visual">
          <geometry><box><size>{segment_length + 0.1} {width + 0.2} 0.1</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Bricks</name></script></material>
        </visual>
      </link>"""

        # --- ДОБАВЛЕННЫЙ БЛОК ПРЕПЯТСТВИЙ ---
        if i % 7 == 0 and i > 0:
            obs_x = next_x
            # Ограничиваем разброс по Y, чтобы препятствие не сливалось со стеной
            obs_y = next_y + random.uniform(-width/4, width/4)
            obs_h = random.uniform(1.0, 2.2) # Высота столба
            
            # Сталагмит (из пола) или сталактит (с потолка)
            if i % 14 == 0: 
                obs_z = FLOOR_Z + obs_h/2
                name_prefix = "stalagmite"
            else:
                obs_z = CEILING_Z - obs_h/2
                name_prefix = "stalactite"

            links += f"""
      <link name="obs_{name_prefix}_{i}">
        <pose>{obs_x} {obs_y} {obs_z} 0 0 {random.uniform(0, 3.14)}</pose>
        <collision name="collision">
          <geometry><box><size>0.4 0.4 {obs_h}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.4 0.4 {obs_h}</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Wood</name></script></material>
        </visual>
      </link>"""
        # ------------------------------------
        
        current_x, current_y = next_x, next_y

    # Тупик в конце
    links += f"""
      <link name="end_wall">
        <pose>{current_x + 0.2} {current_y} 1.5 0 0 {current_angle}</pose>
        <collision name="collision">
          <geometry><box><size>0.2 {width} 3</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>0.2 {width} 3</size></box></geometry>
          <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Bricks</name></script></material>
        </visual>
      </link>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(sdf_header + links + sdf_footer)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_cave(sys.argv[1])
    else:
        print("Usage: python3 procedural_cave.py <output_world_path>")