cmake_minimum_required(VERSION 2.6)
project(nn)

include_directories(include libcl4u/include)

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
set(SOURCE
	${SOURCE}
	source/hw/kit.cpp
	source/hw/buffer.cpp
	source/hw/layer.cpp
	source/hw/conn.cpp
	source/hw/factory.cpp
)
endif(OPENCL)

add_library(${PROJECT_NAME} STATIC ${SOURCE})

if(OPENCL)
add_subdirectory(libcl4u)
target_link_libraries(${PROJECT_NAME} cl4u)
endif(OPENCL)
