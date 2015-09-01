#pragma once

#include <nn/hw/bp/layer.hpp>
#include <nn/bp/layerext.hpp>
#include <nn/hw/layerext.hpp>

template <int X>
class LayerExtHW_BP : public virtual LayerHW_BP, public virtual LayerExtHW<X>, public virtual LayerExt_BP<X>
{
public:
	LayerExtHW_BP() = delete;
	virtual ~LayerExtHW_BP() = default;
};

template <>
class LayerExtHW_BP<LayerFunc::UNIFORM> : public virtual LayerHW_BP, public virtual LayerExtHW<LayerFunc::UNIFORM>, public virtual LayerExt_BP<LayerFunc::UNIFORM>
{
protected:
	LayerExtHW_BP() : LayerExtHW_BP(getID(), getSize()) {}
public:
	LayerExtHW_BP(ID id, int size) : Layer(id, size) {}
	virtual ~LayerExtHW_BP() = default;
};

template <>
class LayerExtHW_BP<LayerFunc::SIGMOID> : 
        public virtual LayerHW_BP, 
        public virtual LayerExtHW<LayerFunc::SIGMOID>, 
        public virtual LayerExt_BP<LayerFunc::SIGMOID>
{
protected:
	LayerExtHW_BP() : LayerExtHW_BP(getID(), getSize()) {}
public:
	LayerExtHW_BP(ID id, int size) : Layer(id, size) {}
	virtual ~LayerExtHW_BP() = default;
	
protected:
	static float _sigma_deriv(float a);
	virtual void _updateError() override;
};

template <>
class LayerExtHW_BP<LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY> : 
        public virtual LayerHW_BP, 
        public virtual LayerExtHW<LayerFunc::SIGMOID>, 
        public virtual LayerExt_BP<LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY>
{
private:
	bool desired = false;
protected:
	LayerExtHW_BP() : LayerExtHW_BP(getID(), getSize()) {}
public:
	LayerExtHW_BP(ID id, int size) : Layer(id, size) {}
	virtual ~LayerExtHW_BP() = default;
	
	virtual float getCost(float *result) const override;
protected:
	static float _sigma_deriv(float a);
	virtual void _updateError() override;
	virtual void _setDesiredOutput(float *result) override;
};
