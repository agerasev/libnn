#include <nn/opencl/factory.hpp>

nn::cl::Factory::Factory(const std::string &kernel_file)
{
	session = new ::cl::session;
	program = new ::cl::program(kernel_file,session->get_context().get_cl_context(),session->get_device_id());
	program->bind_queue(session->get_queue().get_cl_command_queue());
}

nn::cl::Factory::~Factory()
{
	delete program;
	delete session;
}

nn::cl::Layer *nn::cl::Factory::createLayer(Layer::ID id, int size)
{
	nn::cl::Layer *layer = new nn::cl::Layer(id,size,session->get_context().get_cl_context(),program->get_kernel("fill"));
	layer->bindQueue(session->get_queue().get_cl_command_queue());
	return layer;
}

nn::cl::Connection *nn::cl::Factory::createConnection(Connection::ID id, int input_size, int output_size)
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

void nn::cl::Factory::destroyLayer(nn::Layer *layer)
{
	delete layer;
}

void nn::cl::Factory::destroyConnection(nn::Connection *connection)
{
	delete connection;
}
