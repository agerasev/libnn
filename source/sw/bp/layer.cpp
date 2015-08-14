#include <nn/sw/bp/layer.hpp>

LayerSW_BP::LayerSW_BP(ID id, int size)
    : Layer(id, size), LayerSW(id, size)
{
	_input_error = new float[size];
	_output_error = new float[size];
}

LayerSW_BP::~LayerSW_BP()
{
	delete[] _input_error;
	delete[] _output_error;
}

float *LayerSW_BP::getInputError()
{
	return _input_error;
}

const float *LayerSW_BP::getInputError() const
{
	return _input_error;
}

float *LayerSW_BP::getOutputError()
{
	return _output_error;
}

const float *LayerSW_BP::getOutputError() const
{
	return _output_error;
}

float LayerSW_BP::getCost() const
{
	float cost = 0.0f;
	for(int i = 0; i < getSize(); ++i)
	{
		float error = _output_error[i];
		cost += error*error;
	}
	return cost/(2.0f*getSize());
}

void LayerSW_BP::_writeError(const float *sample)
{
	float *output = getOutput();
	for(int i = 0; i < getSize(); ++i)
	{
		_output_error[i] = sample[i] - output[i];
	}
}

void LayerSW_BP::_readError(float *data) const
{
	for(int i = 0; i < getSize(); ++i)
	{
		data[i] = _input_error[i];
	}
}

void LayerSW_BP::_updateError()
{
	float *_temp_error = _input_error;
	_input_error = _output_error;
	_output_error = _temp_error;
}

void LayerSW_BP::_clearError()
{
	for(int i = 0; i < getSize(); ++i)
	{
		_input_error[i] = 0.0f;
	}
}
