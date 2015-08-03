#include <nn/sw/layer.hpp>

LayerSW::LayerSW(ID id, int size)
	: Layer(id, size)
{
	_input_buffer = new float[size];
	_output_buffer = new float[size];
}

LayerSW::~LayerSW()
{
	delete[] _input_buffer;
	delete[] _output_buffer;
}

float *LayerSW::getInput()
{
	return _input_buffer;
}

const float *LayerSW::getInput() const
{
	return _input_buffer;
}

float *LayerSW::getOutput()
{
	return _output_buffer;
}

const float *LayerSW::getOutput() const
{
	return _output_buffer;
}

void LayerSW::_write(const float *data)
{
	for(int i = 0; i < getSize(); ++i)
	{
		_input_buffer[i] = data[i];
	}
}

void LayerSW::_read(float *data) const
{
	for(int i = 0; i < getSize(); ++i)
	{
		data[i] = _output_buffer[i];
	}
}

void LayerSW::_clear()
{
	for(int i = 0; i < getSize(); ++i)
	{
		_input_buffer[i] = 0.0f;
	}
}

void LayerSW::_update()
{
	float *temp_buffer = _input_buffer;
	_input_buffer = _output_buffer;
	_output_buffer = temp_buffer;
}
