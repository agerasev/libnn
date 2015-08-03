#pragma once

#include <nn/network.hpp>

#include <nn/learn/layerplugin.hpp>
#include <nn/learn/connectionplugin.hpp>

#include <map>

namespace nn
{
namespace bp
{
class Network : public nn::Network
{	
public:
	Network();
	virtual ~Network();
	
	virtual void stepForward() override;
	virtual void stepBackward();
};
}
}
