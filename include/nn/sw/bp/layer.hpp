#pragma once

#include <nn/bp/layer.hpp>
#include <nn/sw/layer.hpp>

class LayerSW_BP : public virtual LayerSW, public virtual Layer_BP
{
private:
	float *_input_error;
	float *_output_error;
	
public:
	LayerSW_BP(ID id, int size);
	virtual ~LayerSW_BP();
	
	float *getInputError();
	const float *getInputError() const;
	float *getOutputError();
	const float *getOutputError() const;
	
	virtual float getCost() const override;
	
protected:
	virtual void _writeError(const float *sample) override;
	virtual void _readError(float *data) const override;
	virtual void _updateError() override;
	virtual void _clearError() override;
};
