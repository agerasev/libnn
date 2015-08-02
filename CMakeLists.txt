cmake_minimum_required(VERSION 2.6)
project(libnn)

add_definitions(-DCL_PROFILING)
include_directories(libcl4u include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

set(SOURCE 
	source/layer.cpp 
	source/connection.cpp 
	source/network.cpp
	source/exception.cpp
	source/opencl/layer.cpp
	source/opencl/connection.cpp
	source/opencl/factory.cpp
	source/software/layer.cpp
	source/software/connection.cpp
)

add_library(${PROJECT_NAME} STATIC ${SOURCE})
