#pragma once

#include <map>
#include <functional>

#include "layer.hpp"
#include "conn.hpp"

class Net
{
private:
	std::map<Layer::ID, Layer *> _layers;
	std::map<Conn::ID, Conn *> _conns;
	std::map<Conn::ID, std::pair<Layer::ID, Layer::ID>> _struct;
	
public:
	Net();
	virtual ~Net();
	
	void addLayer(Layer *layer);
	void addConn(Conn *conn, Layer::ID input, Layer::ID output);
	
	Layer *getLayer(Layer::ID id);
	Conn *getConn(Conn::ID id);
	
	void forLayers(std::function<void(Layer *)> func);
	void forConns(std::function<void(Conn *)> func);
	void forConns(std::function<void(Conn *, Layer *, Layer *)> func);
	
	virtual void stepForward();
};
