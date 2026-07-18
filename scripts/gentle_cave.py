"""
Gentle cave generator for curriculum learning Stage 2.
Generates a cave with gentle turns (±30°), 80m long, 5m wide.
Light obstacles every 20 segments, smooth height variations.
"""

import random
import math
import sys


def generate_gentle_cave(output_path, length=100.0, width=5.0, segment_length=1.5):
    """
    Generate a gentle cave SDF file.
    - Gentle turns ±30° (bias ±0.3 rad)
    - Light stalactites every 20 segments
    - Smooth height variations ±0.5m
    - Double walls
    """
    sdf_header = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="gentle_cave_world">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    
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
    <model name="gentle_cave">
      <static>true</static>
"""
    sdf_footer = """
    </model>
  </world>
</sdf>
"""

    wall_thickness = 0.8
    wall_overlap = 1.5
    wall_height = 5.0
    cave_material = "Gazebo/Bricks"
    material_xml = f'<material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>{cave_material}</name></script></material>'

    num_segments = int(length / segment_length)
    links = ""
    link_id = 0

    current_x, current_y, current_angle = 0.0, 0.0, 0.0
    turn_bias = 0.0
    current_floor_z = 0.0
    current_ceiling_z = 3.5

    # Gentle height variations
    height_change_count = random.randint(3, 5)
    height_indices = sorted(random.sample(range(5, num_segments - 8), height_change_count))

    # Light obstacles every 20 segments
    obstacle_interval = 20

    for i in range(num_segments):
        # Gentle turns (±30° = ±0.52 rad, but we use ±0.3 for gentleness)
        turn_bias += random.uniform(-0.12, 0.12)
        turn_bias = max(-0.3, min(0.3, turn_bias))
        current_angle += turn_bias * 0.25 + random.uniform(-0.08, 0.08)
        current_angle = max(-0.52, min(0.52, current_angle))  # ±30°

        # Smooth height variations
        if i in height_indices:
            mode = random.choice(["ceiling_down", "floor_up"])
            if mode == "ceiling_down":
                current_ceiling_z = max(2.5, current_ceiling_z - random.uniform(0.3, 0.5))
            else:
                current_floor_z = min(0.5, current_floor_z + random.uniform(0.2, 0.4))
        else:
            # Gradually return to normal
            current_ceiling_z += (3.5 - current_ceiling_z) * 0.05
            current_floor_z += (0.0 - current_floor_z) * 0.05

        half_width = width / 2.0
        ceil_z = current_ceiling_z

        # Next position
        dx = segment_length * math.cos(current_angle)
        dy = segment_length * math.sin(current_angle)
        next_x = current_x + dx
        next_y = current_y + dy

        # Left wall
        links += f"""
      <link name="wall_left_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y - half_width:.3f} {wall_height / 2.0:.3f} 0 0 {current_angle:.3f}</pose>
        <visual name="visual">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {wall_thickness:.3f} {wall_height:.3f}</size></box></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {wall_thickness:.3f} {wall_height:.3f}</size></box></geometry>
        </collision>
      </link>
"""
        link_id += 1

        # Right wall
        links += f"""
      <link name="wall_right_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y + half_width:.3f} {wall_height / 2.0:.3f} 0 0 {current_angle:.3f}</pose>
        <visual name="visual">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {wall_thickness:.3f} {wall_height:.3f}</size></box></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {wall_thickness:.3f} {wall_height:.3f}</size></box></geometry>
        </collision>
      </link>
"""
        link_id += 1

        # Outer walls (fillers)
        if i > 0:
            links += f"""
      <link name="filler_left_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y - half_width - wall_thickness - 0.4:.3f} {wall_height / 2.0:.3f} 0 0 {current_angle:.3f}</pose>
        <visual name="visual">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {wall_thickness:.3f} {wall_height:.3f}</size></box></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {wall_thickness:.3f} {wall_height:.3f}</size></box></geometry>
        </collision>
      </link>
"""
            link_id += 1

            links += f"""
      <link name="filler_right_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y + half_width + wall_thickness + 0.4:.3f} {wall_height / 2.0:.3f} 0 0 {current_angle:.3f}</pose>
        <visual name="visual">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {wall_thickness:.3f} {wall_height:.3f}</size></box></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {wall_thickness:.3f} {wall_height:.3f}</size></box></geometry>
        </collision>
      </link>
"""
            link_id += 1

        # Ceiling
        links += f"""
      <link name="ceiling_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y:.3f} {ceil_z:.3f} 0 0 {current_angle:.3f}</pose>
        <visual name="visual">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {width + wall_thickness * 2:.3f} {wall_thickness:.3f}</size></box></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {width + wall_thickness * 2:.3f} {wall_thickness:.3f}</size></box></geometry>
        </collision>
      </link>
"""
        link_id += 1

        # Floor
        links += f"""
      <link name="floor_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y:.3f} {current_floor_z - wall_thickness / 2.0:.3f} 0 0 {current_angle:.3f}</pose>
        <visual name="visual">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {width + wall_thickness * 2:.3f} {wall_thickness:.3f}</size></box></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><box><size>{segment_length + wall_overlap:.3f} {width + wall_thickness * 2:.3f} {wall_thickness:.3f}</size></box></geometry>
        </collision>
      </link>
"""
        link_id += 1

        # Light stalactites every 20 segments
        if i > 0 and i % obstacle_interval == 0 and i < num_segments - 5:
            stalactite_x = (current_x + next_x) / 2.0 + random.uniform(-0.5, 0.5)
            stalactite_y = current_y + random.uniform(-half_width * 0.5, half_width * 0.5)
            stalactite_z = ceil_z - 0.3
            stalactite_height = random.uniform(0.3, 0.6)

            links += f"""
      <link name="stalactite_{i}">
        <pose>{stalactite_x:.3f} {stalactite_y:.3f} {stalactite_z:.3f} 0 0 0</pose>
        <visual name="visual">
          <geometry><cylinder><radius>0.15</radius><length>{stalactite_height:.3f}</length></cylinder></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><cylinder><radius>0.15</radius><length>{stalactite_height:.3f}</length></cylinder></geometry>
        </collision>
      </link>
"""
            link_id += 1

        current_x = next_x
        current_y = next_y

    # Dead end wall
    links += f"""
      <link name="dead_end">
        <pose>{current_x + 0.5:.3f} {current_y:.3f} {wall_height / 2.0:.3f} 0 0 {current_angle:.3f}</pose>
        <visual name="visual">
          <geometry><box><size>{wall_thickness:.3f} {width + wall_thickness * 2 + 1.0:.3f} {wall_height:.3f}</size></box></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><box><size>{wall_thickness:.3f} {width + wall_thickness * 2 + 1.0:.3f} {wall_height:.3f}</size></box></geometry>
        </collision>
      </link>
"""

    sdf = sdf_header + links + sdf_footer

    with open(output_path, "w") as f:
        f.write(sdf)

    print(f"[GENTLE CAVE] Generated: {output_path}")
    print(f"  Length: {length}m, Width: {width}m")
    print(f"  Segments: {num_segments}, Links: {link_id + 1}")
    print(f"  Max turn angle: ±30°")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        output = "cave.world"
    else:
        output = sys.argv[1]
    generate_gentle_cave(output)
