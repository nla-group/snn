cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("SNN"
    DESCRIPTION "snn: A lightweight fast exact radius query algorithm"
    VERSION 1.0)

set(INCLUDE_DIR include)
set(SOURCE_DIR src)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_DIR}/*.cpp)

find_library(LAPACK liblapacke.so REQUIRED PATHS /usr/lib/x86_64-linux-gnu/liblapacke.so) # link lapacke
find_library(CBLAS libgslcblas.so REQUIRED PATHS /usr/lib/x86_64-linux-gnu/libgslcblas.so) # link blas


add_library(snn STATIC ${LAPACK} ${CBLAS} ${SRC_FILES})
target_include_directories(snn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR})

add_executable(SNN ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cpp)
target_link_libraries(SNN PUBLIC snn ${LAPACK} ${CBLAS})