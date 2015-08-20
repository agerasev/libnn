#pragma once

#define EXT_NONE       0x0000
#define EXT_SIGMOID    0x0001

#include <nn/layer.hpp>

template <int X>
class LayerExt : public virtual Layer
{
public:
	static const int extension = X;

protected:
	LayerExt() : Layer(0, 0) {}
public:
	virtual ~LayerExt() = default;
};
