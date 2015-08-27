#pragma once

#include <nn/hw/layer.hpp>
#include <nn/layerext.hpp>

template <int X>
class LayerExtHW : public virtual LayerHW, public virtual LayerExt<X>
{
public:
	LayerExtHW() = delete;
	virtual ~LayerExtHW() = default;
};

template <>
class LayerExtHW<LayerFunc::UNIFORM> : public virtual LayerHW, public virtual LayerExt<LayerFunc::UNIFORM>
{
protected:
	LayerExtHW() : LayerExtHW(getID(), getSize(), this) {}
public:
	LayerExtHW(ID id, int size, const KitHW *kit);
	virtual ~LayerExtHW() = default;
};

template <>
class LayerExtHW<LayerFunc::SIGMOID> : public virtual LayerHW, public virtual LayerExt<LayerFunc::SIGMOID>
{
protected:
	LayerExtHW() : LayerExtHW(getID(), getSize(), this) {}
public:
	LayerExtHW(ID id, int size, const KitHW *kit);
	virtual ~LayerExtHW() = default;

protected:
	virtual void _update() override;
};
