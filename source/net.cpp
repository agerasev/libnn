#include <nn/net.hpp>

#include <nn/exception.hpp>

Net::Net()
{
	
}

Net::~Net()
{
	
}

void Net::addLayer(Layer *layer)
{
	_layers.insert(std::pair<Layer::ID, Layer *>(layer->getID(), layer));
}

void Net::addConn(Conn *conn, Layer::ID input, Layer::ID output)
{
	if(_layers.find(input) == _layers.end())
		throw Exception(std::string("no layer with id=") + std::to_string(input) + std::string(" in network"));
	if(_layers.find(output) == _layers.end())
		throw Exception(std::string("no layer with id=") + std::to_string(output) + std::string(" in network"));
	
	_conns.insert(std::pair<Conn::ID, Conn *>(conn->getID(), conn));
	_struct.insert(
				std::pair<Conn::ID, std::pair<Layer::ID, Layer::ID> >(
					conn->getID(), 
					std::pair<Layer::ID, Layer::ID>(input, output)
					)
				);
}

Layer *Net::getLayer(Layer::ID id)
{
	return _layers[id];
}

Conn *Net::getConn(Conn::ID id)
{
	return _conns[id];
}

void Net::forLayers(std::function<void(Layer *)> func)
{
	for(const std::pair<Layer::ID, Layer *> &p : _layers)
		func(p.second);
}

void Net::forConns(std::function<void(Conn *)> func)
{
	for(const std::pair<Conn::ID, Conn *> &p : _conns)
	{
		func(p.second);
	}
}

void Net::forConnsWithLayers(std::function<void(Conn *, Layer *, Layer *)> func)
{
	for(const std::pair<Conn::ID, Conn *> &p : _conns)
	{
		std::pair<Layer::ID, Layer::ID> lp = _struct[p.first];
		func(p.second,_layers[lp.first],_layers[lp.second]);
	}
}

void Net::stepForward()
{
	for(std::pair<Layer::ID, Layer *> p : _layers)
	{
		p.second->update();
	}
	for(const std::pair<Conn::ID, std::pair<Layer::ID, Layer::ID> > &p : _struct)
	{
		_conns[p.first]->transmit(_layers[p.second.first],_layers[p.second.second]);
	}
}
