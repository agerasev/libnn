#include <nn/hw/bp/factory.hpp>

#include <nn/exception.hpp>

FactoryHW_BP::FactoryHW_BP(const std::string &kernel_file)
  : FactoryHW(kernel_file)
{
	
}

LayerHW_BP *FactoryHW_BP::newLayer(Layer::ID id, int size, int extension)
{
	LayerHW_BP *layer;
	KitHW kit(&session->get_context(), &program->get_kernel_map());
	if(extension == LayerFunc::UNIFORM)
	{
		layer = new LayerHW_BP(id, size, &kit);
	}
	/*
	else
	if(extension == LayerFunc::SIGMOID)
	{
		layer = new LayerExtHW_BP<LayerFunc::SIGMOID>(id, size, &kit);
	}
	*/
	else
	{
		throw Exception("there is no such layer extension code: " + std::to_string(extension));
	}
	layer->bindQueue(&session->get_queue());
	return layer;
}

ConnHW_BP *FactoryHW_BP::newConn(Conn::ID id, int input_size, int output_size)
{
	KitHW kit(&session->get_context(), &program->get_kernel_map());
	ConnHW_BP *connection = new ConnHW_BP(id, input_size, output_size, &kit);
	connection->bindQueue(&session->get_queue());
	return connection;
}

