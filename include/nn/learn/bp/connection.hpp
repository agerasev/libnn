#pragma once

#include <nn/learn/bp/layer.hpp>

namespace nn
{
namespace bp
{
class ConnectionPlugin
{
public:
	ConnectionPlugin() = default;
	virtual ~ConnectionPlugin() = default;
	
	virtual void backpropagate(nn::LayerX<LayerPlugin> *to, const nn::LayerX<LayerPlugin> *from) = 0;
	virtual void addGradient() = 0;
	virtual void clearCradient() = 0;
};
}
}
