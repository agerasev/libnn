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
class LayerExtSW_BP<LayerFunc::UNIFORM> : public virtual LayerSW_BP, public virtual LayerExtSW<LayerFunc::UNIFORM>, public virtual LayerExt_BP<LayerFunc::UNIFORM>
{
protected:
	LayerExtSW_BP() : LayerExtSW_BP(getID(), getSize()) {}
public:
	LayerExtSW_BP(ID id, int size) : Layer(id, size) {}
	virtual ~LayerExtSW_BP() = default;
};

template <>
class LayerExtSW_BP<LayerFunc::SIGMOID> : 
        public virtual LayerSW_BP, 
        public virtual LayerExtSW<LayerFunc::SIGMOID>, 
        public virtual LayerExt_BP<LayerFunc::SIGMOID>
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

template <>
class LayerExtSW_BP<LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY> : 
        public virtual LayerSW_BP, 
        public virtual LayerExtSW<LayerFunc::SIGMOID>, 
        public virtual LayerExt_BP<LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY>
{
private:
	bool desired = false;
protected:
	LayerExtSW_BP() : LayerExtSW_BP(getID(), getSize()) {}
public:
	LayerExtSW_BP(ID id, int size) : Layer(id, size) {}
	virtual ~LayerExtSW_BP() = default;
	
	virtual float getCost(float *result) const override;
protected:
	static float _sigma_deriv(float a);
	virtual void _updateError() override;
	virtual void _setDesiredOutput(float *result) override;
};
