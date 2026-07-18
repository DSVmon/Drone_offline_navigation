"""
Straight cave generator for curriculum learning Stage 1.
Generates a simple straight tunnel 50m long, 6m wide, no turns.
Includes obstacles (beams and pillars) for obstacle avoidance training.
"""

import math
import random
import sys


def generate_straight_cave(output_path, length=100.0, width=6.0, height=3.5, segment_length=1.5):
    """
    Generate a simple straight cave SDF file.
    - Straight tunnel, no turns
    - No obstacles (stalactites/stalagmites)
    - No height variations
    - Double walls (0.8m thickness)
    - Dead end wall at the end
    """
    sdf_header = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="straight_cave_world">
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
    <model name="straight_cave">
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

    current_x = 0.0
    current_y = 0.0

    for i in range(num_segments):
        # No turns - straight tunnel
        next_x = current_x + segment_length
        next_y = current_y

        half_width = width / 2.0
        ceil_z = height

        # Inner left wall
        links += f"""
      <link name="wall_left_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y - half_width:.3f} {wall_height / 2.0:.3f} 0 0 0</pose>
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

        # Inner right wall
        links += f"""
      <link name="wall_right_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y + half_width:.3f} {wall_height / 2.0:.3f} 0 0 0</pose>
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

        # Outer left wall (filler)
        if i > 0:
            links += f"""
      <link name="filler_left_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y - half_width - wall_thickness - 0.4:.3f} {wall_height / 2.0:.3f} 0 0 0</pose>
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

            # Outer right wall (filler)
            links += f"""
      <link name="filler_right_{i}">
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y + half_width + wall_thickness + 0.4:.3f} {wall_height / 2.0:.3f} 0 0 0</pose>
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
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y:.3f} {ceil_z:.3f} 0 0 0</pose>
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
        <pose>{(current_x + next_x) / 2.0:.3f} {current_y:.3f} {-wall_thickness / 2.0:.3f} 0 0 0</pose>
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

        current_x = next_x
        current_y = next_y

    # --- OBSTACLES: beams and pillars ---
    obstacle_interval = 10  # obstacle every 10 segments (~15m)
    for i in range(0, num_segments, obstacle_interval):
        if i < 3 or i > num_segments - 5:
            continue  # skip near start and end

        obs_x = i * segment_length + segment_length / 2.0

        # Horizontal beam (from left wall to right wall at random height)
        beam_height = random.uniform(1.0, 2.5)  # height of beam center
        beam_thickness = 0.3
        beam_width = width - 1.0  # leave gap on sides

        links += f"""
      <link name="beam_{i}">
        <pose>{obs_x:.3f} {current_y:.3f} {beam_height:.3f} 0 0 0</pose>
        <visual name="visual">
          <geometry><box><size>{beam_width:.3f} {beam_thickness:.3f} {beam_thickness:.3f}</size></box></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><box><size>{beam_width:.3f} {beam_thickness:.3f} {beam_thickness:.3f}</size></box></geometry>
        </collision>
      </link>
"""
        link_id += 1

        # Vertical pillar (from floor to ceiling at random position)
        pillar_y = current_y + random.uniform(-width/2.5, width/2.5)
        pillar_width = 0.3

        links += f"""
      <link name="pillar_{i}">
        <pose>{obs_x:.3f} {pillar_y:.3f} {height/2.0:.3f} 0 0 0</pose>
        <visual name="visual">
          <geometry><box><size>{pillar_width:.3f} {pillar_width:.3f} {height:.3f}</size></box></geometry>
          {material_xml}
        </visual>
        <collision name="collision">
          <geometry><box><size>{pillar_width:.3f} {pillar_width:.3f} {height:.3f}</size></box></geometry>
        </collision>
      </link>
"""
        link_id += 1

    # Dead end wall at the end
    links += f"""
      <link name="dead_end">
        <pose>{current_x + 0.5:.3f} {current_y:.3f} {wall_height / 2.0:.3f} 0 0 0</pose>
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

    print(f"[STRAIGHT CAVE] Generated: {output_path}")
    print(f"  Length: {length}m, Width: {width}m, Height: {height}m")
    print(f"  Segments: {num_segments}, Links: {link_id + 1}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        output = "cave.world"
    else:
        output = sys.argv[1]
    generate_straight_cave(output)
