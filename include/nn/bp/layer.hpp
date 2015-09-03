#pragma once

#include <nn/layer.hpp>

class Layer_BP : public virtual Layer
{
protected:
	Layer_BP() : Layer(getID(), getSize()) {}
public:
	Layer_BP(ID id, int size) : Layer(id, size) {}
	virtual ~Layer_BP() = default;
	
	virtual Layer::Buffer &getInputError() = 0;
	virtual Layer::Buffer &getOutputError() = 0;
	virtual const Layer::Buffer &getInputError() const = 0;
	virtual const Layer::Buffer &getOutputError() const = 0;
	
	void updateError();
	void setDesiredOutput(const float *result);
	virtual float getCost(const float *result) const = 0;
	
protected:
	virtual void _updateError() = 0;
	virtual void _setDesiredOutput(const float *result) = 0;
};
