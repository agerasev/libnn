#pragma once

#include <string>

#include <cl/session.hpp>
#include <cl/program.hpp>

#include <nn/opencl/layer.hpp>
#include <nn/opencl/connection.hpp>

namespace nn
{
namespace cl
{
class Factory
{
private:
	::cl::session *session;
	::cl::program *program;
	
public:
	Factory(const std::string &kernel_file);
	~Factory();
	
	nn::cl::Layer *createLayer(Layer::ID id, int size);
	nn::cl::Connection *createConnection(Connection::ID id, int input_size, int output_size);
	
	void destroyLayer(nn::Layer *layer);
	void destroyConnection(nn::Connection *connection);
};
}
}
