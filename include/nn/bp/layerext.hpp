#pragma once

#include <nn/bp/layer.hpp>
#include <nn/layerext.hpp>

template <int X>
class LayerExt_BP : public virtual Layer_BP, public virtual LayerExt<X>
{
protected:
	LayerExt_BP() : LayerExt_BP(getID(), getSize()) {}
public:
	LayerExt_BP(ID id, int size) : Layer(id, size) {}
	virtual ~LayerExt_BP() = default;
};
