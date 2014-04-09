cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

project(opencl-option-pricer)

set(OPENCL_LIBS "ocl")
set(STDCL_LIBS "stdcl")

include(FindPkgConfig)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB OPENCL_KERNELS kernels/*.cl)
file(GLOB OPENCL_KERNEL_HEADERS kernels/*.h)

add_custom_command(OUTPUT opencl_kernels.o COMMAND clcc -I../kernels/ -c -o opencl_kernels.o -c ${OPENCL_KERNELS} DEPENDS ${OPENCL_KERNELS} ${OPENCL_KERNEL_HEADERS})

add_executable(opencl_option_pricer src/main.cpp opencl_kernels.o)
target_link_libraries(opencl_option_pricer ${OPENCL_LIBS} ${STDCL_LIBS})

add_executable(random_test src/random.cpp opencl_kernels.o)
target_link_libraries(random_test ${OPENCL_LIBS} ${STDCL_LIBS})

add_executable(monte_carlo src/monte_carlo.cpp src/matrix.cpp src/statistics.cpp src/statistics_opencl.cpp src/opencl_utils.cpp opencl_kernels.o)
target_link_libraries(monte_carlo ${OPENCL_LIBS} ${STDCL_LIBS})
