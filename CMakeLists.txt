cmake_minimum_required(VERSION 2.6)
project(libnn)

add_definitions(-DCL_PROFILING)
include_directories(libcl4u)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
file(GLOB SOURCE "*.cpp" "*.c")

target_link_libraries(${PROJECT_NAME} OpenCL)
add_library(libnn STATIC ${SOURCE})
