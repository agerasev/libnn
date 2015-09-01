#include <nn/hw/bp/layer.hpp>

LayerHW_BP::LayerHW_BP(ID id, int size, const KitHW *kit)
    : Layer(id, size), KitHW(kit), _input_error(size), _output_error(size)
{
	
}

LayerHW::BufferHW &LayerHW_BP::getInputError()
{
	return _input_error;
}

LayerHW::BufferHW &LayerHW_BP::getOutputError()
{
	return _output_error;
}

const LayerHW::BufferHW &LayerHW_BP::getInputError() const
{
	return _input_error;
}

const LayerHW::BufferHW &LayerHW_BP::getOutputError() const
{
	return _output_error;
}

void LayerHW_BP::_setDesiredOutput(const float *result)
{
	cl::work_range range(getSize());
	getOutputError().getBuffer()->store_data(result);
	getKernel("setErrorC_quartic")->evaluate(
	      range, getSize(), getOutput().getBuffer(), getOutputError().getBuffer()
	      );
}

void LayerHW_BP::_setDesiredOutput(const cl::buffer_object *result)
{
	cl::work_range range(getSize());
	getKernel("setError_quartic")->evaluate(
	      range, getSize(), result, getOutput().getBuffer(), getOutputError().getBuffer()
	      );
}

float LayerHW_BP::getCost(float *result) const
{
	float sum = 0.0f;
	const int size = getSize();
	float *data = new float[size];
	get
	const float *output = getOutput().getData();
	for(int i = 0; i < size; ++i)
	{
		float dif = output[i] - result[i];
		sum += dif*dif;
	}
	sum *= 0.5f;
	return sum;
}

void LayerHW_BP::_updateError()
{
	const int size = getSize();
	float *output = getInputError().getData();
	const float *input = getOutputError().getData();
	for(int i = 0; i < size; ++i)
	{
		output[i] = input[i];
	}
}
