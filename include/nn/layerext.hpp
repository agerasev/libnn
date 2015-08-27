#pragma once

namespace LayerFunc
{
static const int 
UNIFORM = 0x00,
SIGMOID = 0x01;
}

namespace LayerCost
{
static const int
QUADRIC       = 0x0000,
CROSS_ENTROPY = 0x0100;
}

#include <nn/layer.hpp>

template <int X>
class LayerExt : public virtual Layer
{
public:
	static const int extension = X;

protected:
	LayerExt() : Layer(getID(), getSize()) {}
public:
	virtual ~LayerExt() = default;
};
