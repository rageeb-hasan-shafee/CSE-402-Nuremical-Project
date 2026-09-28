cmake_minimum_required(VERSION 3.16)
project(nbody_openmp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NBODY_NATIVE "Tune for this CPU (-march=native / -mcpu=native)" OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)
set(COMMON_INC ${CMAKE_CURRENT_SOURCE_DIR}/../../common/cpp)

add_executable(nbody_serial nbody_omp.cpp)
add_executable(nbody_omp    nbody_omp.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(NATIVE_FLAG -mcpu=native)
else()
  set(NATIVE_FLAG -march=native)
endif()

foreach(t nbody_serial nbody_omp)
  target_include_directories(${t} PRIVATE ${COMMON_INC})
  if(NBODY_NATIVE AND NOT MSVC)
    target_compile_options(${t} PRIVATE ${NATIVE_FLAG})
  endif()
endforeach()

target_link_libraries(nbody_omp PRIVATE OpenMP::OpenMP_CXX)
