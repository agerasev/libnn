#pragma once

#include <map>
#include <functional>

#include "layer.hpp"
#include "connection.hpp"

namespace nn
{
class Network
{
private:
	std::map<Layer::ID, Layer *> _layers;
	std::map<Connection::ID, Connection *> _conns;
	std::map<Connection::ID, std::pair<Layer::ID, Layer::ID>> _struct;
	
public:
	Network();
	virtual ~Network();
	
	void addLayer(Layer *layer);
	void addConnection(Connection *conn, Layer::ID input, Layer::ID output);
	
	Layer *getLayer(Layer::ID id);
	Connection *getConnecion(Connection::ID id);
	
	void forLayers(std::function<void(Layer *)> func);
	void forConnections(std::function<void(Connection *)> func);
	void forConnections(std::function<void(Connection *, Layer *, Layer *)> func);
	
	virtual void stepForward();
};
}
