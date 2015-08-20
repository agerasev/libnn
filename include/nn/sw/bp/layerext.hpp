#pragma once

#include <nn/sw/bp/layer.hpp>
#include <nn/bp/layerext.hpp>
#include <nn/sw/layerext.hpp>

template <int X>
class LayerExtSW_BP : public virtual LayerSW_BP, public virtual LayerExtSW<X>, public virtual LayerExt_BP<X>
{
public:
	LayerExtSW_BP() = delete;
	virtual ~LayerExtSW_BP() = default;
};

template <>
class LayerExtSW_BP<EXT_NONE> : public virtual LayerSW_BP, public virtual LayerExtSW<EXT_NONE>, public virtual LayerExt_BP<EXT_NONE>
{
protected:
	LayerExtSW_BP() : LayerExtSW_BP(getID(), getSize()) {}
public:
	LayerExtSW_BP(ID id, int size) : Layer(id, size) {}
	virtual ~LayerExtSW_BP() = default;
};

template <>
class LayerExtSW_BP<EXT_SIGMOID> : public virtual LayerSW_BP, public virtual LayerExtSW<EXT_SIGMOID>, public virtual LayerExt_BP<EXT_SIGMOID>
{
protected:
	LayerExtSW_BP() : LayerExtSW_BP(getID(), getSize()) {}
public:
	LayerExtSW_BP(ID id, int size) : Layer(id, size) {}
	virtual ~LayerExtSW_BP() = default;
	
protected:
	static float _sigma_deriv(float a);
	virtual void _updateError() override;
};
