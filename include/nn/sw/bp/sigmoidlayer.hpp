#pragma once

#include <nn/sw/sigmoidlayer.hpp>
#include <nn/sw/bp/layer.hpp>

class SigmoidLayerSW_BP : public virtual SigmoidLayerSW, public virtual LayerSW_BP
{
public:
	SigmoidLayerSW_BP(ID id, int size);
	virtual ~SigmoidLayerSW_BP();
	
protected:
	virtual void _updateError() override;
	// virtual void _updateError_CE(); // For cross-entropy cost function
};
