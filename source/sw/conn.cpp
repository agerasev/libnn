#include <nn/sw/conn.hpp>

#include <nn/exception.hpp>

ConnSW::ConnSW(ID id, int input_size, int output_size, int weight_size, int bias_size)
	: Conn(id, input_size, output_size), _weight(weight_size), _bias(bias_size)
{
	
}

ConnSW::ConnSW() : ConnSW(getID(), getInputSize(), getOutputSize())
{
	
}

ConnSW::ConnSW(ID id, int input_size, int output_size)
	: ConnSW(id, input_size, output_size, input_size*output_size, output_size)
{
	
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
	
	const float *weight = _weight.getData();
	const float *bias = _bias.getData();
	for(int j = 0; j < out_size; ++j)
	{
		float sum = 0.0;
		for(int i = 0; i < in_size; ++i)
		{
			sum += input[i]*weight[in_size*j + i];
		}
		output[j] += sum + bias[j];
	}
}

ConnSW::BufferSW &ConnSW::getWeight()
{
	return _weight;
}

ConnSW::BufferSW &ConnSW::getBias()
{
	return _bias;
}

const ConnSW::BufferSW &ConnSW::getWeight() const
{
	return _weight;
}
const ConnSW::BufferSW &ConnSW::getBias() const
{
	return _bias;
}
