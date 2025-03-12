cmake_minimum_required(VERSION 3.8)
project(opt_control)

set(CMAKE_CXX_STANDARD 20)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} include)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(test_newton_find_root
  src/test_newton_find_root.cpp)
target_include_directories(test_newton_find_root PRIVATE include)
target_compile_definitions(test_newton_find_root PRIVATE -DAUTODIFF_EIGEN_FOUND)

add_executable(test_newton_minimize
  src/test_newton_minimize.cpp)
target_include_directories(test_newton_minimize PRIVATE include)
target_compile_definitions(test_newton_minimize PRIVATE -DAUTODIFF_EIGEN_FOUND)

add_executable(test_newton_equality_minimize
  src/test_newton_equality_minimize.cpp)
target_include_directories(test_newton_equality_minimize PRIVATE include)
target_compile_definitions(test_newton_equality_minimize PRIVATE -DAUTODIFF_EIGEN_FOUND)

install (TARGETS
  test_newton_find_root
  test_newton_minimize
  test_newton_equality_minimize
DESTINATION lib/${PROJECT_NAME})

ament_package()
