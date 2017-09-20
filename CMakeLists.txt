cmake_minimum_required(VERSION 2.8.3)
project(general_particle_filter LANGUAGES CXX CUDA)



find_package(catkin REQUIRED COMPONENTS
 roscpp
 roslint
)

find_package(CUDA)

catkin_python_setup()

add_compile_options(-std=c++11)

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES particle_filter_cpu particle_filter_gpu
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

target_compile_options(particle_filter_cpu PRIVATE -std=gnu++11)

add_library(planar_object_cpu
        src/example/planar_object_cpu.cpp
        src/example/cpu_example_pf.cpp)

target_compile_options(planar_object_cpu PRIVATE -std=gnu++11)

add_executable(planar_object_cpu_demo
        src/example/cpu_example.cpp
        )

target_compile_options(planar_object_cpu_demo PRIVATE -std=gnu++11)

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
    if ($ENV{TRAVIS})

    endif()
endif()

if(${CUDA_FOUND} AND  CMAKE_VERSION VERSION_GREATER 3.8)
    if($ENV{TRAVIS})
        message("On Travis, not building the cuda programs")
    else()
        message("Building CUDA programs and tests...")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=compute_52")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=sm_60")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_61,code=sm_61")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_62,code=sm_62")

        add_library(particle_filter_gpu 
            src/gpu/particle_filter.cu
            src/gpu/particle_filter_kernels.cu
        )
        target_compile_options(particle_filter_gpu PUBLIC -std=c++11)


        add_executable(main_gpu src/gpu/main.cu)
        target_link_libraries(particle_filter_gpu
            ${CUDA_LIBRARIES}
            ${catkin_LIBRARIES}
        )
        
        target_link_libraries(main_gpu particle_filter_gpu)
        target_compile_options(main_gpu PUBLIC -std=c++11)

        add_library(planar_object_gpu
                src/example/planar_object_gpu.cu
                src/example/gpu_example_pf.cu
                )
        target_compile_options(planar_object_gpu PUBLIC -std=c++11)

        add_executable(planar_object_gpu_demo src/example/gpu_example.cu)
        target_compile_options(planar_object_gpu_demo PUBLIC -std=c++11)

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

        #add_library(test_wrapper
        #        test/gpu/test_wrapper.cpp
        #        )

        #target_link_libraries(test_wrapper
        #        particle_filter_gpu
        #        ${catkin_LIBRARIES}
        #        ${CUDA_LIBRARIES})

    endif()
endif()

