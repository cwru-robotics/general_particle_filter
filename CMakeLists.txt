cmake_minimum_required(VERSION 2.8.3)
project(general_particle_filter)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp)
find_package(CUDA)


catkin_python_setup()

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES particle_filter_cpu
   LIBRARIES particle_filter_gpu
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



if(CUDA_FOUND)
    message("...Found CUDA ${CUDA_VERSION}")
    if(OpenCV_CUDA_VERSION)
        message("...Opencv compiled with cuda version ${OpenCV_CUDA_VERSION}")
    endif()
endif()

if(CUDA_FOUND AND OpenCV_CUDA_VERSION)
    SET(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
endif()

if(CUDA_FOUND AND OpenCV_CUDA_VERSION)
    cuda_add_library(particle_filter_gpu src/gpu/particle_filter.cu)

    cuda_add_executable(main_gpu src/gpu/main.cu)

    target_link_libraries(particle_filter_gpu ${CUDA_LIBRARIES} ${catkin_LIBRARIES} )
    target_link_libraries(main_gpu particle_filter_gpu)

    catkin_add_gtest(gtest_gpu_resampling test/gpu/resample_test.cpp)
    target_link_libraries(gtest_gpu_resampling
            particle_filter_gpu
            ${catkin_LIBRARIES}
            ${CUDA_LIBRARIES})
endif()
