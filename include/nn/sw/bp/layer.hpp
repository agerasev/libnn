#pragma once

#include <nn/bp/layer.hpp>
#include <nn/sw/layer.hpp>

class LayerSW_BP : public virtual LayerSW, public virtual Layer_BP
{
private:
	LayerSW::BufferSW _input_error;
	LayerSW::BufferSW _output_error;
	
protected:
	LayerSW_BP() : LayerSW_BP(getID(), getSize()) {}
public:
	LayerSW_BP(ID id, int size);
	virtual ~LayerSW_BP() = default;
	
	virtual LayerSW::BufferSW &getInputError() override;
	virtual LayerSW::BufferSW &getOutputError() override;
	virtual const LayerSW::BufferSW &getInputError() const override;
	virtual const LayerSW::BufferSW &getOutputError() const override;
	
	virtual float getCost() const override;
	
protected:
	virtual void _updateError() override;
};
