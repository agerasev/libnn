#include <nn/sw/bp/conn.hpp>
#include <nn/sw/bp/layer.hpp>
#include <nn/exception.hpp>

ConnSW_BP::ConnSW_BP(ID id, int input_size, int output_size)
    : Conn(id, input_size, output_size), _weight_grad(input_size*output_size), _bias_grad(output_size)
{
	
}

ConnSW::BufferSW &ConnSW_BP::getWeightGrad()
{
	return _weight_grad;
}

ConnSW::BufferSW &ConnSW_BP::getBiasGrad()
{
	return _bias_grad;
}

const ConnSW::BufferSW &ConnSW_BP::getWeightGrad() const
{
	return _weight_grad;
}

const ConnSW::BufferSW &ConnSW_BP::getBiasGrad() const
{
	return _bias_grad;
}

void ConnSW_BP::_commitGrad(float delta)
{
	float *weight = getWeight().getData();
	float *bias = getBias().getData();
	const float *weight_grad = getWeightGrad().getData();
	const float *bias_grad = getBiasGrad().getData();
	const int weight_size = getWeight().getSize();
	const int bias_size = getBias().getSize();
	const float norm = delta/getBPCount();
	for(int i = 0; i < weight_size; ++i)
	{
		weight[i] -= weight_grad[i]*norm;
	}
	for(int i = 0; i < bias_size; ++i)
	{
		bias[i] -= bias_grad[i]*norm;
	}
	getWeightGrad().clear();
	getBiasGrad().clear();
}

void ConnSW_BP::_backprop(const Layer *to, const Layer_BP *from)
{
	const LayerSW *to_sw = dynamic_cast<const LayerSW *>(to);
	if(to_sw == nullptr)
		throw Exception("output layer is not derived from LayerSW");
	
	const LayerSW_BP *from_sw = dynamic_cast<const LayerSW_BP *>(from);
	if(from_sw == nullptr)
		throw Exception("input layer is not derived from LayerSW_BP");
	
	const float *input_error = from_sw->getOutputError().getData();
	int sx = getInputSize(), sy = getOutputSize();
	
	float *weight_grad = getWeightGrad().getData();
	float *bias_grad = getBiasGrad().getData();
	for(int i = 0; i < sy; ++i)
	{
		bias_grad[i] = input_error[i];
	}
	
	const float *output = to_sw->getOutput().getData();
	for(int iy = 0; iy < sy; ++iy)
	{
		for(int ix = 0; ix < sx; ++ix)
		{
			weight_grad[iy*sx + ix] += output[ix]*input_error[iy];
		}
	}
}

void ConnSW_BP::_backprop(Layer_BP *to, const Layer_BP *from)
{
	LayerSW_BP *to_sw = dynamic_cast<LayerSW_BP *>(to);
	if(to_sw == nullptr)
		throw Exception("output layer is not derived from LayerSW_BP");
	
	const LayerSW_BP *from_sw = dynamic_cast<const LayerSW_BP *>(from);
	if(from_sw == nullptr)
		throw Exception("input layer is not derived from LayerSW_BP");
	
	float *output_error = to_sw->getInputError().getData();
	const float *input_error = from_sw->getOutputError().getData();
	const float *weight = getWeight().getData();
	int sx = getInputSize(), sy = getOutputSize();
	for(int ix = 0; ix < sx; ++ix)
	{
		for(int iy = 0; iy < sy; ++iy)
		{
			// TODO: fence in opencl to optimize cache
			output_error[ix] += weight[iy*sx + ix]*input_error[iy];
		}
	}
	
	_backprop(static_cast<const Layer *>(to), from);
}
