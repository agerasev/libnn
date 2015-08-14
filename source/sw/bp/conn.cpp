#include <nn/sw/bp/conn.hpp>
#include <nn/sw/bp/layer.hpp>
#include <nn/exception.hpp>

ConnSW_BP::ConnSW_BP(ID id, int input_size, int output_size)
    : Conn(id, input_size, output_size), ConnSW(id, input_size, output_size)
{
	_weight_grad = new float[_weight_size];
	_bias_grad = new float[_bias_size];
	_bp_count = 0;
}

ConnSW_BP::~ConnSW_BP()
{
	delete[] _weight_grad;
	delete[] _bias_grad;
}

void ConnSW_BP::addGrad()
{
	float *weight = getWeight();
	float *bias = getBias();
	float norm_factor = 1.0f/_bp_count;
	for(int i = 0; i < _weight_size; ++i)
	{
		weight[i] += norm_factor*_weight_grad[i];
	}
	for(int i = 0; i < _bias_size; ++i)
	{
		bias[i] += norm_factor*_bias_grad[i];
	}
}

void ConnSW_BP::clearGrad()
{
	for(int i = 0; i < _weight_size; ++i)
	{
		_weight_grad[i] = 0.0f;
	}
	for(int i = 0; i < _bias_size; ++i)
	{
		_bias_grad[i] = 0.0f;
	}
}

void ConnSW_BP::backprop(Layer *to, const Layer *from, float delta)
{
	LayerSW_BP *output = dynamic_cast<LayerSW_BP *>(to);
	LayerSW_BP *input = dynamic_cast<const LayerSW_BP *>(from);
	if(input == nullptr)
		throw Exception("input layer is not derived from LayerSW_BP");
	
	...
}
