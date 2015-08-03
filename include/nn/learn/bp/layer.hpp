#pragma once

#include <nn/layer.hpp>

namespace nn
{
namespace bp
{
class LayerPlugin
{
public:
	LayerPlugin();
	virtual ~LayerPlugin() = default;
	
	virtual void updateError() = 0;
	virtual void clearError() = 0;
};
}
}
