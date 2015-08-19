#include <nn/sw/conn.hpp>

#include <nn/exception.hpp>

ConnSW::ConnSW(ID id, int input_size, int output_size, int weight_size, int bias_size)
	: Conn(id, input_size, output_size)
{
	_weight_size = weight_size;
	_bias_size = bias_size;
	_weight_buffer = new float[_weight_size];
	_bias_buffer = new float[_bias_size];
}

ConnSW::ConnSW() : ConnSW(getID(), getInputSize(), getOutputSize())
{
	
}

ConnSW::ConnSW(ID id, int input_size, int output_size)
	: ConnSW(id, input_size, output_size, input_size*output_size, output_size)
{
	
}

ConnSW::~ConnSW()
{
	delete[] _weight_buffer;
	delete[] _bias_buffer;
}

float *ConnSW::getWeight()
{
	return _weight_buffer;
}

const float *ConnSW::getWeight() const
{
	return _weight_buffer;
}

float *ConnSW::getBias()
{
	return _bias_buffer;
}

const float *ConnSW::getBias() const
{
	return _bias_buffer;
}

void ConnSW::_transmit(const Layer *from, Layer *to) const
{
	const LayerSW *sw_from = dynamic_cast<const LayerSW *>(from);
	if(sw_from == nullptr)
		throw Exception("input layer is not derived from LayerSW");
	
	LayerSW *sw_to = dynamic_cast<LayerSW *>(to);
	if(sw_to == nullptr)
		throw Exception("output layer is not derived from LayerSW");
	
	const float *input = sw_from->getOutput().getData();
	float *output = sw_to->getInput().getData();
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

void ConnSW::readWeight(float *data) const
{
	for(int i = 0; i < _weight_size; ++i)
	{
		data[i] = _weight_buffer[i];
	}
}

void ConnSW::readBias(float *data) const
{
	for(int i = 0; i < _bias_size; ++i)
	{
		data[i] = _bias_buffer[i];
	}
}

void ConnSW::writeWeight(const float *data)
{
	for(int i = 0; i < _weight_size; ++i)
	{
		_weight_buffer[i] = data[i];
	}
}

void ConnSW::writeBias(const float *data)
{
	for(int i = 0; i < _bias_size; ++i)
	{
		_bias_buffer[i] = data[i];
	}
}

int ConnSW::getWeightSize() const
{
	return _weight_size;
}

int ConnSW::getBiasSize() const
{
	return _bias_size;
}
