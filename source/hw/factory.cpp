#include <nn/hw/factory.hpp>
#include <nn/exception.hpp>

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

LayerHW *FactoryHW::newLayer(Layer::ID id, int size, int extension)
{
	LayerHW *layer;
	KitHW kit(&session->get_context(), &program->get_kernel_map(), &session->get_queue());
	if(extension == LayerFunc::UNIFORM)
	{
		layer = new LayerHW(id, size, &kit);
	}
	else
	if(extension == LayerFunc::SIGMOID)
	{
		layer = new LayerExtHW<LayerFunc::SIGMOID>(id, size, &kit);
	}
	else
	{
		throw Exception("there is no such layer extension code: " + std::to_string(extension));
	}
	// layer->bindQueue(&session->get_queue());
	return layer;
}

ConnHW *FactoryHW::newConn(Conn::ID id, int input_size, int output_size)
{
	KitHW kit(&session->get_context(), &program->get_kernel_map(), &session->get_queue());
	ConnHW *connection = new ConnHW(id, input_size, output_size, &kit);
	// connection->bindQueue(&session->get_queue());
	return connection;
}
