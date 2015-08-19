#pragma once

#include <nn/bp/layer.hpp>
#include <nn/layerext.hpp>

template <int X>
class LayerExt_BP : public virtual Layer_BP, public virtual LayerExt
{
public:
	LayerExt_BP() : Layer(0,0) {}
	virtual ~LayerExt_BP() = default;
};
