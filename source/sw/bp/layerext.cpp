#include <nn/sw/bp/layerext.hpp>

#include <cmath>

float LayerExtSW_BP<EXT_SIGMOID>::_sigma_deriv(float a)
{
	float e = exp(a);
	float d = (1.0f + e);
	return e/(d*d);
}

void LayerExtSW_BP<EXT_SIGMOID>::_updateError()
{
	const int size = getSize();
	float *output_error = getOutputError().getData();
	const float *input_error = getInputError().getData();
	const float *input = getInput().getData();
	for(int i = 0; i < size; ++i)
	{
		output_error[i] = input_error[i]*_sigma_deriv(input[i]);
	}
}
