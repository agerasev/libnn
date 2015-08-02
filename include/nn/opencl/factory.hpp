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
	Factory(const std::string &kernel_file)
	{
		session = new ::cl::session;
		program = new ::cl::program(kernel_file,session->get_context().get_cl_context(),session->get_device_id());
		program->bind_queue(session->get_queue().get_cl_command_queue());
	}
	
	~Factory()
	{
		delete program;
		delete session;
	}
	
	nn::cl::Layer *createLayer(Layer::ID id, int size)
	{
		nn::cl::Layer *layer = new nn::cl::Layer(id,size,session->get_context().get_cl_context(),program->get_kernel("fill"));
		layer->bindQueue(session->get_queue().get_cl_command_queue());
		return layer;
	}
	
	nn::cl::Connection *createConnection(Connection::ID id, int input_size, int output_size)
	{
		nn::cl::Connection *connection = new 
		    nn::cl::Connection(
		      id, input_size, output_size,
		      program->get_kernel("full_product"),
		      session->get_context().get_cl_context()
		      );
		connection->bindQueue(session->get_queue().get_cl_command_queue());
		return connection;
	}
	
	void destroyLayer(nn::Layer *layer)
	{
		delete layer;
	}
	
	void destroyConnection(nn::Connection *connection)
	{
		delete connection;
	}
};
}
}
