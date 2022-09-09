cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("SNN"
    DESCRIPTION "snn: A lightweight fast exact radius query algorithm"
    VERSION 1.0)

set(INCLUDE_DIR include)

set(CMAKE_MODULE_PATH cmake)
set(SOURCE_DIR src)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_DIR}/*.cpp)

find_package(OpenMP)
find_package(LAPACK)
find_package(LAPACKE) # link lapacke
find_package(BLAS) # link blas

add_library(snn STATIC ${SRC_FILES})


target_include_directories(snn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR})
target_link_libraries(snn PUBLIC ${BLAS_LIBRARIES} ${LAPACKE_LIBRARIES})

add_executable(SNN ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cpp)

if(OpenMP_CXX_FOUND) #openMP
    target_link_libraries(SNN PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(SNN PUBLIC snn ${BLAS_LIBRARIES}  ${LAPACKE_LIBRARIES})
