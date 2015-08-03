#pragma once

#include <string>

#include <cl/session.hpp>
#include <cl/program.hpp>

#include <nn/hw/layer.hpp>
#include <nn/hw/conn.hpp>

class FactoryHW
{
private:
	cl::session *session;
	cl::program *program;
	
public:
	FactoryHW(const std::string &kernel_file);
	~FactoryHW();
	
	LayerHW *createLayer(Layer::ID id, int size);
	ConnHW *createConn(Conn::ID id, int input_size, int output_size);
};
