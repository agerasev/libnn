#pragma once

#include <nn/learn/connectionplugin.hpp>

namespace nn
{
namespace sw
{
namespace bp
{
class ConnectionPlugin : public nn::bp::ConnectionPlugin
{
protected:
	float *_weight_gradient = nullptr;
	float *_bias_gradient = nullptr;
	int _bp_count = 0;
	
	virtual void _init();
	virtual void _cleanup();
	
public:
	ConnectionPlugin()
	{
		_init();
	}
	
	virtual ~ConnectionPlugin()
	{
		_cleanup();
	}
};
}
}
}
