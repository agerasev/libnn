#include <nn/hw/bp/layerext.hpp>

#include <cmath>

void LayerExtHW_BP<LayerFunc::SIGMOID>::_updateError()
{
	getKernel("updateError_sigmoid")->evaluate(
		cl::work_range(getSize()), getSize(),
		getInputError().getBuffer(), getOutputError().getBuffer(), 
		getInput().getBuffer()
		);
}

float LayerExtHW_BP<LayerFunc::SIGMOID|LayerCost::CROSS_ENTROPY>::getCost(const float *result) const
{
	float sum = 0.0f;
	const int size = getSize();
	float *output = new float[size];
	getOutput().getBuffer()->load_data(output);
	for (int i = 0; i < size; ++i)
	{
		float dc = result[i]*log(output[i]) + (1.0f - result[i])*log(1.0f - output[i]);
		if(std::isnan(dc) || std::isinf(dc))
			continue;
		sum -= dc;
	}
	delete[] output;
	return sum;
}

// TODO: Add non-optimized version

void LayerExtHW_BP<LayerFunc::SIGMOID|LayerCost::CROSS_ENTROPY>::_updateError()
{
	if(desired)
	{
		getKernel("copy")->evaluate(
			cl::work_range(getSize()), getSize(),
			getOutputError().getBuffer(), getInputError().getBuffer()
			);
	}
	else
	{
		getKernel("updateError_sigmoid_crossEntropy")->evaluate(
			cl::work_range(getSize()), getSize(),
			getInputError().getBuffer(), getOutputError().getBuffer(), 
			getInput().getBuffer(), getOutput().getBuffer()
			);
	}
	desired = false;
}

void LayerExtHW_BP<LayerFunc::SIGMOID|LayerCost::CROSS_ENTROPY>::_setDesiredOutput(const float *result)
{
	getOutputError().getBuffer()->store_data(result);
	getKernel("setErrorReuse")->evaluate(
		cl::work_range(getSize()), getSize(),
		getOutput().getBuffer(), getOutputError().getBuffer()
		);
	desired = true;
}

void LayerExtHW_BP<LayerFunc::SIGMOID|LayerCost::CROSS_ENTROPY>::_setDesiredOutput(const cl::buffer_object *result)
{
	getKernel("setError")->evaluate(
		cl::work_range(getSize()), getSize(),
		result, getOutput().getBuffer(), getOutputError().getBuffer()
		);
	desired = true;
}
