cmake_minimum_required(VERSION 3.24)

project(CUDAPractice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(cuda_kernels OBJECT
  kernels/kernels.cu
)

target_include_directories(cuda_kernels
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(cuda_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES native
)

add_executable(cuda_practice
  main.cu
  launch/launch.cu
  $<TARGET_OBJECTS:cuda_kernels>
)

target_include_directories(cuda_practice
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cuda_practice
  PRIVATE
    CUDA::cudart
)

set_target_properties(cuda_practice PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES native
)

add_executable(kernels_correctness_vs_cublas
  tests/kernels_correctness_vs_cublas.cu
  $<TARGET_OBJECTS:cuda_kernels>
)

target_include_directories(kernels_correctness_vs_cublas
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kernels_correctness_vs_cublas
  PRIVATE
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(kernels_correctness_vs_cublas PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES native
)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME kernels_correctness_vs_cublas
           COMMAND kernels_correctness_vs_cublas)
endif()

