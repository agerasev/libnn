#pragma once

#include <nn/network.hpp>

#include <nn/learn/bp/layer.hpp>
#include <nn/learn/bp/connection.hpp>

#include <map>

namespace nn
{
namespace bp
{
class Network : public nn::Network
{
private:
	std::map<nn::Layer::ID, nn::LayerX<LayerPlugin> *> _bp_layers;
	std::map<nn::Connection::ID, nn::ConnectionX<ConnectionPlugin> *> _bp_conns;
	std::map<nn::Connection::ID, std::pair<nn::Layer::ID, nn::Layer::ID>> _bp_struct;
	
public:
	Network();
	virtual ~Network();
	
	void addLearnLayer(nn::LayerX<LayerPlugin> *layer);
	void addLearnConnection(nn::ConnectionX<ConnectionPlugin> *conn, Layer::ID input, Layer::ID output);
	
	Layer *getLearnLayer(Layer::ID id);
	Connection *getLearnConnection(Connection::ID id);
	
	void forLearnLayers(std::function<void(nn::LayerX<LayerPlugin> *)> func);
	void forLearnConnections(std::function<void(nn::ConnectionX<ConnectionPlugin> *)> func);
	void forLearnConnections(std::function<void(nn::ConnectionX<ConnectionPlugin> *, nn::LayerX<LayerPlugin> *, nn::LayerX<LayerPlugin> *)> func);
	
	virtual void stepForward() override;
	virtual void stepBackward();
};
}
}
