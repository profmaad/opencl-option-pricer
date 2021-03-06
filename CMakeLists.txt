cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

project(opencl-option-pricer)

set(OPENCL_LIBS "ocl")
set(STDCL_LIBS "stdcl")
set(JSONCPP_LIBS "jsoncpp")

include_directories("/usr/local/browndeer/include")
link_directories("/usr/local/browndeer/lib")

include_directories("/usr/include/jsoncpp" "/usr/local/include/jsoncpp")

include(FindPkgConfig)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB OPENCL_KERNELS kernels/*.cl)
file(GLOB OPENCL_KERNEL_HEADERS kernels/*.h)

add_custom_command(OUTPUT opencl_kernels.o COMMAND clcc -I${CMAKE_CURRENT_SOURCE_DIR}/kernels/ -c -o opencl_kernels.o -c ${OPENCL_KERNELS} DEPENDS ${OPENCL_KERNELS} ${OPENCL_KERNEL_HEADERS})

set(opencl_option_pricer_SOURCES src/main.cpp src/matrix.cpp src/statistics.cpp src/statistics_opencl.cpp src/opencl_utils.cpp src/kernels.cpp src/json_helper.cpp src/json_outputter.cpp
  src/opencl_option.cpp src/closed_form_opencl_option.cpp src/european_opencl_option.cpp src/asian_geometric_opencl_option.cpp src/basket_geometric_opencl_option.cpp src/monte_carlo_opencl_option.cpp src/asian_arithmetic_opencl_option.cpp src/basket_arithmetic_opencl_option.cpp src/options.cpp)

add_executable(opencl_option_pricer ${opencl_option_pricer_SOURCES} opencl_kernels.o)
target_link_libraries(opencl_option_pricer ${OPENCL_LIBS} ${STDCL_LIBS} ${JSONCPP_LIBS})

add_executable(random_test src/random.cpp src/statistics_opencl.cpp opencl_kernels.o)
target_link_libraries(random_test ${OPENCL_LIBS} ${STDCL_LIBS})

add_executable(random_test_host src/random_host.cpp src/statistics_opencl.cpp opencl_kernels.o)
target_link_libraries(random_test_host ${OPENCL_LIBS} ${STDCL_LIBS})
