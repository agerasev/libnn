#include <nn/hw/bp/layer.hpp>

LayerHW_BP::LayerHW_BP(ID id, int size, const KitHW *kit)
    : Layer(id, size), KitHW(kit), _input_error(size, kit), _output_error(size, kit)
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

void LayerHW_BP::setDesiredOutput(const cl::buffer_object *result)
{
	_setDesiredOutput(result);
	getOutputError().validate(true);
}

void LayerHW_BP::_setDesiredOutput(const float *result)
{
	getOutputError().getBuffer()->store_data(result);
	getKernel("setErrorC")->evaluate(
	      cl::work_range(getSize()), getSize(), 
	      getOutput().getBuffer(), getOutputError().getBuffer()
	      );
}

void LayerHW_BP::_setDesiredOutput(const cl::buffer_object *result)
{
	getKernel("setError")->evaluate(
	      cl::work_range(getSize()), getSize(), 
	      result, getOutput().getBuffer(), getOutputError().getBuffer()
	      );
}

float LayerHW_BP::getCost(const float *result) const
{
	float sum = 0.0f;
	const int size = getSize();
	float *output = new float[size];
	getOutput().getBuffer()->load_data(output);
	for(int i = 0; i < size; ++i)
	{
		float dif = output[i] - result[i];
		sum += dif*dif;
	}
	delete[] output;
	sum *= 0.5f;
	return sum;
}

void LayerHW_BP::_updateError()
{
	getKernel("updateError")->evaluate(
	      cl::work_range(getSize()), getSize(),
	      getInputError().getBuffer(), getOutputError().getBuffer()
	      );
}

void LayerHW_BP::_bindQueue(cl::queue *queue)
{
	getInputError().bindQueue(queue);
	getOutputError().bindQueue(queue);
	LayerHW::_bindQueue(queue);
}
