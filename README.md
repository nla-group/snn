cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(SNN VERSION 1.0)

set(MAIN_DIR snn)
set(SOURCE_DIR src)


file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN_DIR}/${SOURCE_DIR}/*.cpp)

find_library(LAPACK liblapacke.so REQUIRED PATHS /usr/lib/x86_64-linux-gnu/liblapacke.so) # link lapacke
find_library(CBLAS libgslcblas.so REQUIRED PATHS /usr/lib/x86_64-linux-gnu/libgslcblas.so) # link blas

add_library(SNN STATIC ${SRC_FILES})
target_include_directories(SNN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN_DIR}/include)

add_executable(AAA ${CMAKE_CURRENT_SOURCE_DIR}/${MAIN_DIR}/testheader.cpp)
target_link_libraries(AAA PUBLIC SNN ${MYLIB})