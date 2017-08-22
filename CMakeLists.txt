cmake_minimum_required(VERSION 2.8.3)
project(general_particle_filter)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
 roscpp
 roslint
)
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

add_library(planar_object_cpu
        src/example/planar_object_cpu.cpp
        src/example/cpu_example_pf.cpp)

add_executable(planar_object_cpu_demo
        src/example/cpu_example.cpp
        )

target_link_libraries(planar_object_cpu_demo
        planar_object_cpu
        particle_filter_cpu
        ${catkin_LIBRARIES}
        )

catkin_add_gtest(gtest_cpu_resampling test/cpu/resample_test.cpp)

target_link_libraries(gtest_cpu_resampling
       particle_filter_cpu
       ${catkin_LIBRARIES})


file(GLOB_RECURSE SRC_FILES_CPP src/*.cpp)
file(GLOB_RECURSE TEST_FILES_CPP test/*.cpp)
file(GLOB_RECURSE INCLUDE_FILES_H include/*.h)
roslint_cpp(${SRC_FILES_CPP} ${TEST_FILES_CPP} ${INCLUDE_FILES_H})


file(GLOB_RECURSE SRC_FILES_PY src/*.py)
file(GLOB_RECURSE TEST_FILES_PY test/*.py)
roslint_python(setup.py ${SRC_FILES_PY} ${TEST_FILES_PY})


roslint_add_test()

if(CUDA_FOUND)
    message("...Found CUDA ${CUDA_VERSION}")
endif()

if(CUDA_FOUND)
    set(CUDA_VERBOSE_BUILD ON)
    # "--compiler-options --relocatable-device-code=true --compile"
    # set(CUDA_NVCC_FLAGS "-arch=compute_50" CACHE STRING "nvcc flags" FORCE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11  -arch=sm_61")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_20")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=compute_52")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=sm_61")
    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_62,code=sm_62")
endif()

if(CUDA_FOUND)
    cuda_add_library(particle_filter_gpu
            src/gpu/particle_filter.cu
            src/gpu/particle_filter_kernels.cu
            )

    cuda_add_executable(main_gpu src/gpu/main.cu)

    target_link_libraries(particle_filter_gpu ${CUDA_LIBRARIES} ${catkin_LIBRARIES} )
    target_link_libraries(main_gpu particle_filter_gpu)


    cuda_add_library(planar_object_gpu
            src/example/planar_object_gpu.cu
            src/example/gpu_example_pf.cu
            )

    cuda_add_executable(planar_object_gpu_demo src/example/gpu_example.cu)


    target_link_libraries(planar_object_gpu
            particle_filter_gpu
            ${catkin_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${CUDA_curand_LIBRARY}
            )

    target_link_libraries(planar_object_gpu_demo
            planar_object_gpu
            particle_filter_gpu
            ${catkin_LIBRARIES}
            ${CUDA_LIBRARIES}
            ${CUDA_curand_LIBRARY}
            )

    if ($ENV{TRAVIS})
        message("On Travis, not building the cuda unit tests")
    else()
        message("Not on Travis, building the cuda unit tests")
        catkin_add_gtest(gtest_gpu_resampling test/gpu/resample_test.cpp)
        target_link_libraries(gtest_gpu_resampling
                particle_filter_gpu
                ${catkin_LIBRARIES}
                ${CUDA_LIBRARIES})
    endif()
endif()

