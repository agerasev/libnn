#include <nn/network.hpp>

#include <nn/exception.hpp>

nn::Network::Network()
{
	
}

nn::Network::~Network()
{
	
}

void nn::Network::addLayer(Layer *layer)
{
	_layers.insert(std::pair<Layer::ID, Layer *>(layer->getID(), layer));
}

void nn::Network::addConnection(Connection *conn, Layer::ID input, Layer::ID output)
{
	if(_layers.find(input) == _layers.end())
		throw Exception(std::string("no layer with id=") + std::to_string(input) + std::string(" in network"));
	if(_layers.find(output) == _layers.end())
		throw Exception(std::string("no layer with id=") + std::to_string(output) + std::string(" in network"));
	
	_conns.insert(std::pair<Connection::ID, Connection *>(conn->getID(), conn));
	_struct.insert(
				std::pair<Connection::ID, std::pair<Layer::ID, Layer::ID> >(
					conn->getID(), 
					std::pair<Layer::ID, Layer::ID>(input, output)
					)
				);
}

nn::Layer *nn::Network::getLayer(Layer::ID id)
{
	return _layers[id];
}

nn::Connection *nn::Network::getConnecion(Connection::ID id)
{
	return _conns[id];
}

void nn::Network::forLayers(std::function<void(Layer *)> func)
{
	for(const std::pair<Layer::ID, Layer *> &p : _layers)
		func(p.second);
}

void nn::Network::forConnections(std::function<void(Connection *)> func)
{
	for(const std::pair<Connection::ID, Connection *> &p : _conns)
	{
		func(p.second);
	}
}

void nn::Network::forConnections(std::function<void(Connection *, Layer *, Layer *)> func)
{
	for(const std::pair<Connection::ID, Connection *> &p : _conns)
	{
		std::pair<Layer::ID, Layer::ID> lp = _struct[p.first];
		func(p.second,_layers[lp.first],_layers[lp.second]);
	}
}

void nn::Network::stepForward()
{
	for(std::pair<Layer::ID, Layer *> p : _layers)
	{
		p.second->update();
		p.second->clear();
	}
	for(const std::pair<Connection::ID, std::pair<Layer::ID, Layer::ID> > &p : _struct)
	{
		_conns[p.first]->feedforward(_layers[p.second.first],_layers[p.second.second]);
	}
}
