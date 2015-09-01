#pragma once

#include <nn/bp/layer.hpp>
#include <nn/hw/layer.hpp>

class LayerHW_BP : public virtual LayerHW, public virtual Layer_BP
{
private:
	LayerHW::BufferHW _input_error;
	LayerHW::BufferHW _output_error;
	
protected:
	LayerHW_BP() : LayerHW_BP(getID(), getSize(), static_cast<const KitHW *>(this)) {}
public:
	LayerHW_BP(ID id, int size, const KitHW *kit);
	virtual ~LayerHW_BP() = default;
	
	virtual LayerHW::BufferHW &getInputError() override;
	virtual LayerHW::BufferHW &getOutputError() override;
	virtual const LayerHW::BufferHW &getInputError() const override;
	virtual const LayerHW::BufferHW &getOutputError() const override;
	
	virtual float getCost(float *result) const override;
	
protected:
	virtual void _updateError() override;
	virtual void _setDesiredOutput(float *result) override;
};
