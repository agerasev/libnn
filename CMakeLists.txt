cmake_minimum_required(VERSION 2.6)
project(libnn)

include_directories(include libcl4u/include)

set(SOURCE 
	source/layer.cpp 
	source/conn.cpp 
	source/net.cpp
	source/exception.cpp
	source/sw/layer.cpp
	source/sw/conn.cpp
	source/sw/sigmoidlayer.cpp
)

if(OPENCL)
set(SOURCE
	${SOURCE}
	source/hw/layer.cpp
	source/hw/conn.cpp
	source/hw/factory.cpp
)
endif(OPENCL)

add_library(${PROJECT_NAME} STATIC ${SOURCE})

add_subdirectory(libcl4u)
target_link_libraries(${PROJECT_NAME} libcl4u)
