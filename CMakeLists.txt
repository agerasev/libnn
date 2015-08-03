cmake_minimum_required(VERSION 2.6)
project(libnn)

include_directories(libcl4u include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

set(SOURCE 
	source/layer.cpp 
	source/connection.cpp 
	source/network.cpp
	source/exception.cpp
	source/software/layer.cpp
	source/software/connection.cpp
	source/software/sigmoidlayer.cpp
)

find_package(OpenCL)
if(OPENCL_FOUND)
set(SOURCE
	${SOURCE}
	source/opencl/layer.cpp
	source/opencl/connection.cpp
	source/opencl/factory.cpp
)
endif(OPENCL_FOUND)

add_library(${PROJECT_NAME} STATIC ${SOURCE})
