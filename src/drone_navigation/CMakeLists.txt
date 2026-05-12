cmake_minimum_required(VERSION 3.8)
project(drone_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_geometry REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Include directories
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Executables
add_executable(navigation_node src/navigation_node.cpp)
ament_target_dependencies(navigation_node 
  rclcpp 
  std_msgs 
  nav_msgs 
  geometry_msgs 
  sensor_msgs 
  message_filters 
  cv_bridge 
  image_geometry
  tf2
  tf2_geometry_msgs
)
target_link_libraries(navigation_node ${OpenCV_LIBS})

install(TARGETS
  navigation_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
