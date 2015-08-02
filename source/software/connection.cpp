#include <nn/software/connection.hpp>

#include <nn/exception.hpp>

nn::sw::Connection::Connection(ID id, int input_size, int output_size, int weight_buffer_size, int bias_buffer_size)
	: nn::Connection(id, input_size, output_size)
{
	_weight_buffer_size = weight_buffer_size;
	_bias_buffer_size = bias_buffer_size;
	_weight_buffer = new float[_weight_buffer_size];
	_bias_buffer = new float[_bias_buffer_size];
}

nn::sw::Connection::Connection(ID id, int input_size, int output_size)
	: Connection(id, input_size, output_size, input_size*output_size, output_size)
{
	
}

nn::sw::Connection::~Connection()
{
	delete[] _weight_buffer;
	delete[] _bias_buffer;
}

float *nn::sw::Connection::getWeightBuffer()
{
	return _weight_buffer;
}

const float *nn::sw::Connection::getWeightBuffer() const
{
	return _weight_buffer;
}

float *nn::sw::Connection::getBiasBuffer()
{
	return _bias_buffer;
}

const float *nn::sw::Connection::getBiasBuffer() const
{
	return _bias_buffer;
}

void nn::sw::Connection::_feedforward(const nn::Layer *from, nn::Layer *to) const
{
	const nn::sw::Layer *sw_from = dynamic_cast<const nn::sw::Layer *>(from);
	if(sw_from == nullptr)
		throw Exception("input layer is not derived from nn::sw::Layer");
	
	nn::sw::Layer *sw_to = dynamic_cast<nn::sw::Layer *>(to);
	if(sw_to == nullptr)
		throw Exception("output layer is not derived from nn::sw::Layer");
	
	const float *input = sw_from->getOutputBuffer();
	float *output = sw_to->getInputBuffer();
	const int out_size = getOutputSize();
	const int in_size = getInputSize();
	
	for(int j = 0; j < out_size; ++j)
	{
		float sum = 0.0;
		for(int i = 0; i < in_size; ++i)
		{
			sum += input[i]*_weight_buffer[in_size*j + i];
		}
		output[j] += sum + _bias_buffer[j];
	}
}

void nn::sw::Connection::read_weight(float *data) const
{
	for(int i = 0; i < _weight_buffer_size; ++i)
	{
		data[i] = _weight_buffer[i];
	}
}

void nn::sw::Connection::read_bias(float *data) const
{
	for(int i = 0; i < _bias_buffer_size; ++i)
	{
		data[i] = _bias_buffer[i];
	}
}

void nn::sw::Connection::write_weight(const float *data)
{
	for(int i = 0; i < _weight_buffer_size; ++i)
	{
		_weight_buffer[i] = data[i];
	}
}

void nn::sw::Connection::write_bias(const float *data)
{
	for(int i = 0; i < _bias_buffer_size; ++i)
	{
		_bias_buffer[i] = data[i];
	}
}
