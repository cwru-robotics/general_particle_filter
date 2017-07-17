cmake_minimum_required(VERSION 2.8.3)
project(general_particle_filter)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp)
find_package(CUDA)

catkin_python_setup()

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES particle_filter_cpu
   CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

add_library(particle_filter_cpu
   src/cpu/particle_filter.cpp
)

catkin_add_gtest(gtest_cpu_resampling test/cpu/resample_test.cpp)

target_link_libraries(gtest_cpu_resampling
       particle_filter_cpu
       ${catkin_LIBRARIES})