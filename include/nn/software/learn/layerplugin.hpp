#pragma once

#include <nn/learn/layerplugin.hpp>

namespace nn 
{
namespace sw
{
namespace bp
{
class LayerPlugin : public nn::bp::LayerPlugin
{
protected:
	float *_input_error = nullptr;
	float *_output_error = nullptr;
	
	virtual void _init();
	virtual void _cleanup();
	
public:
	LayerPlugin()
	{
		_init();
	}

	virtual ~LayerPlugin()
	{
		_cleanup();
	}
};
}
}
}
