cmake_minimum_required(VERSION 3.8)
project(ddpg_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(ddpg_turtlebot3 src/ddpg_turtlebot3.cpp)
target_include_directories(ddpg_turtlebot3 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(ddpg_turtlebot3 PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS ddpg_turtlebot3
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS  
  ddpg_ros2/tb3_gym_env.py  
  ddpg_ros2/respawn_goal.py
  DESTINATION lib/${PROJECT_NAME} )

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
