#include <nn/sw/bp/layerext.hpp>

#include <cmath>
static float _sigma_deriv_frame(float e)
{
	float d = (1.0f + e);
	return e/(d*d);
}

static float _sigma_deriv(float a)
{
	return _sigma_deriv_frame(exp(a));
}

float LayerExtSW_BP<LayerFunc::SIGMOID>::_sigma_deriv(float a)
{
	return ::_sigma_deriv(a);
}

float LayerExtSW_BP< LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY >::_sigma_deriv(float a)
{
	return ::_sigma_deriv(a);
}

void LayerExtSW_BP<LayerFunc::SIGMOID>::_updateError()
{
	const int size = getSize();
	float *input_error = getInputError().getData();
	const float *output_error = getOutputError().getData();
	const float *input = getInput().getData();
	for(int i = 0; i < size; ++i)
	{
		input_error[i] = output_error[i]*_sigma_deriv(input[i]);
	}
}

float LayerExtSW_BP< LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY >::getCost(float *result) const
{
	float sum = 0.0f;
	const int size = getSize();
	const float *output = getOutput().getData();
	for(int i = 0; i < size; ++i)
	{
		float dc = result[i]*log(output[i]) + (1.0f - result[i])*log(1.0f - output[i]);
		if(std::isnan(dc) || std::isinf(dc))
			continue;
		sum -= dc;
	}
	return sum;
}


#ifdef NN_NO_OPTIM

void LayerExtSW_BP< LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY >::_updateError()
{
	const int size = getSize();
	float *input_error = getInputError().getData();
	const float *output_error = getOutputError().getData();
	const float *input = getInput().getData();
	for(int i = 0; i < size; ++i)
	{
		input_error[i] = output_error[i]*_sigma_deriv(input[i]);
	}
}

void LayerExtSW_BP< LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY >::_setDesiredOutput(float *result)
{
	const int size = getSize();
	float *output_error = getOutputError().getData();
	const float *output = getOutput().getData();
	for(int i = 0; i < size; ++i)
	{
		input_error[i] = result[i]/output[i] - (1.0 - result[i])/(1.0 - output[i]);
	}
}

#else // NN_NO_OPTIM

void LayerExtSW_BP< LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY >::_updateError()
{
	const int size = getSize();
	float *input_error = getInputError().getData();
	const float *output_error = getOutputError().getData();
	if(desired)
	{
		for(int i = 0; i < size; ++i)
		{
			input_error[i] = output_error[i];
		}
	}
	else
	{
		const float *output = getOutput().getData();
		for(int i = 0; i < size; ++i)
		{
			input_error[i] = output_error[i]*output[i]*(1.0 - output[i]);
		}
	}
	desired = false;
}

void LayerExtSW_BP< LayerFunc::SIGMOID | LayerCost::CROSS_ENTROPY >::_setDesiredOutput(float *result)
{
	const int size = getSize();
	float *output_error = getOutputError().getData();
	const float *output = getOutput().getData();
	for(int i = 0; i < size; ++i)
	{
		output_error[i] = result[i] - output[i];
	}
	desired = true;
}

#endif // NN_NO_OPTIM
