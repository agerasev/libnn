#include <nn/hw/factory.hpp>

FactoryHW::FactoryHW(const std::string &kernel_file)
{
	session = new ::cl::session;
	program = new ::cl::program(kernel_file,session->get_context().get_cl_context(),session->get_device_id());
	program->bind_queue(session->get_queue().get_cl_command_queue());
}

FactoryHW::~FactoryHW()
{
	delete program;
	delete session;
}

LayerHW *FactoryHW::createLayer(Layer::ID id, int size)
{
	LayerHW *layer = new LayerHW(id,size,session->get_context().get_cl_context(),program->get_kernel_map());
	layer->bindQueue(session->get_queue());
	return layer;
}

ConnHW *FactoryHW::createConn(Conn::ID id, int input_size, int output_size)
{
	ConnHW *connection = new 
			ConnHW(
				id, input_size, output_size,
	      session->get_context().get_cl_context(),
				program->get_kernel_map()
				);
	connection->bindQueue(session->get_queue());
	return connection;
}
