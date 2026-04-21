cmake_minimum_required(VERSION 3.8)
project(drone_simulation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  worlds
  urdf
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
