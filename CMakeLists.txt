cmake_minimum_required(VERSION 2.6)
project(nn)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall")
elseif (MSVC)
	
endif()

include_directories(include)
include_directories(libla)

set(SOURCE 
	source/buffer.cpp
	source/layer.cpp 
	source/conn.cpp 
	source/net.cpp
	source/exception.cpp
	source/sw/buffer.cpp
	source/sw/layer.cpp
	source/sw/conn.cpp
	source/sw/layerext.cpp
	source/bp/conn.cpp
	source/bp/layer.cpp
	source/bp/net.cpp
	source/sw/bp/conn.cpp
	source/sw/bp/layer.cpp
	source/sw/bp/layerext.cpp
)

if(OPENCL)
	include_directories(libcl4u/include)
	set(SOURCE
		${SOURCE}
		source/hw/kit.cpp
		source/hw/buffer.cpp
		source/hw/layer.cpp
		source/hw/layerext.cpp
		source/hw/conn.cpp
		source/hw/factory.cpp
		source/hw/bp/factory.cpp
		source/hw/bp/layer.cpp
		source/hw/bp/layerext.cpp
		source/hw/bp/conn.cpp
	)
endif(OPENCL)

add_library(${PROJECT_NAME} STATIC ${SOURCE})

if(OPENCL)
	add_subdirectory(libcl4u)
	target_link_libraries(${PROJECT_NAME} cl4u)
endif(OPENCL)
