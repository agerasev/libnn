#pragma once

#include <nn/learn/layerplugin.hpp>

namespace nn
{
namespace bp
{
class ConnectionPlugin
{
public:
	ConnectionPlugin() = default;
	virtual ~ConnectionPlugin() = default;
	
	virtual void backpropagate(nn::Layer *to, const nn::Layer *from) = 0;
	virtual void addGradient() = 0;
	virtual void clearCradient() = 0;
};
}
}
