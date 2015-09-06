#pragma once

#include <string>

#include <cl/session.hpp>
#include <cl/program.hpp>

#include <nn/hw/layerext.hpp>
#include <nn/hw/conn.hpp>

class FactoryHW
{
private:
	cl::session *session;
	cl::program *program;
	
public:
	FactoryHW(const std::string &kernel_file = std::string("libnn/opencl/kernel.c"));
	virtual ~FactoryHW();
	
	virtual LayerHW *newLayer(Layer::ID id, int size, int extension = LayerFunc::UNIFORM);
	virtual ConnHW *newConn(Conn::ID id, int input_size, int output_size);
	
	cl::session *getSession();
	cl::program *getProgram();
	
	const cl::session *getSession() const;
	const cl::program *getProgram() const;
};
