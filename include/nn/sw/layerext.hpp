#pragma once

#include <nn/sw/layer.hpp>
#include <nn/layerext.hpp>

template <int X>
class LayerExtSW : public virtual LayerSW, public virtual LayerExt<X>
{
public:
	LayerExtSW() = delete;
	virtual ~LayerExtSW() = default;
};

template <>
class LayerExtSW<LayerFunc::UNIFORM> : public virtual LayerSW, public virtual LayerExt<LayerFunc::UNIFORM>
{
protected:
	LayerExtSW() : LayerExtSW(getID(), getSize()) {}
public:
	LayerExtSW(ID id, int size);
	virtual ~LayerExtSW() = default;
};

template <>
class LayerExtSW<LayerFunc::SIGMOID> : public virtual LayerSW, public virtual LayerExt<LayerFunc::SIGMOID>
{
protected:
	LayerExtSW() : LayerExtSW(getID(), getSize()) {}
public:
	LayerExtSW(ID id, int size);
	virtual ~LayerExtSW() = default;

protected:
	static float _sigma(float a);
	virtual void _update() override;
};
