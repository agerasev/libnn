#include <nn/sw/bp/layer.hpp>

LayerSW_BP::LayerSW_BP(ID id, int size)
    : Layer(id, size), _input_error(size), _output_error(size)
{
	
}

LayerSW::BufferSW &LayerSW_BP::getInputError()
{
	return _input_error;
}

LayerSW::BufferSW &LayerSW_BP::getOutputError()
{
	return _output_error;
}

const LayerSW::BufferSW &LayerSW_BP::getInputError() const
{
	return _input_error;
}

const LayerSW::BufferSW &LayerSW_BP::getOutputError() const
{
	return _output_error;
}

float LayerSW_BP::getCost() const
{
	float sum = 0.0f;
	const int size = getSize();
	const float *error = _output_error.getData();
	for(int i = 0; i < size; ++i)
	{
		sum += error[i]*error[i];
	}
	sum *= 0.5f;
	return sum;
}

void LayerSW_BP::_updateError()
{
	const int size = getSize();
	float *output = getOutputError().getData();
	const float *input = getInputError().getData();
	for(int i = 0; i < size; ++i)
	{
		output[i] = input[i];
	}
}
