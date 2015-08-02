#include <nn/software/layer.hpp>

nn::sw::Layer::Layer(ID id, int size)
	: nn::Layer(id, size)
{
	_input_buffer = new float[size];
	_output_buffer = new float[size];
}

nn::sw::Layer::~Layer()
{
	delete[] _input_buffer;
	delete[] _output_buffer;
}

float *nn::sw::Layer::getInputBuffer()
{
	return _input_buffer;
}

const float *nn::sw::Layer::getOutputBuffer() const
{
	return _output_buffer;
}

void nn::sw::Layer::_write(const float *data)
{
	for(int i = 0; i < getSize(); ++i)
	{
		_input_buffer[i] = data[i];
	}
}

void nn::sw::Layer::_read(float *data) const
{
	for(int i = 0; i < getSize(); ++i)
	{
		data[i] = _output_buffer[i];
	}
}

void nn::sw::Layer::_clear()
{
	for(int i = 0; i < getSize(); ++i)
	{
		_input_buffer[i] = 0.0f;
	}
}

void nn::sw::Layer::_update()
{
	float *temp_buffer = _input_buffer;
	_input_buffer = _output_buffer;
	_output_buffer = temp_buffer;
}
