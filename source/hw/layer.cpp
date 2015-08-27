#include <nn/hw/layer.hpp>

LayerHW::LayerHW(cl::context context, cl::map<cl::kernel *> *kernels)
	: LayerHW(getID(), getSize(), context, &getKernelMap())
{
	
}

LayerHW::LayerHW(ID id, int size, cl::context context, cl::map<cl::kernel *> *kernels)
	: Layer(id, size), _input(size, context, kernels), _output(size, context, kernels), KernelMapHW(kernels)
{
	
}

LayerHW::~LayerHW()
{
	
}

void LayerHW::_bindQueue(cl_command_queue queue)
{
	_input.bindQueue(queue);
	_output.bindQueue(queue);
}

LayerHW::BufferHW &LayerHW::getInput()
{
	return _input;
}

LayerHW::BufferHW &LayerHW::getOutput()
{
	return _output;
}

const LayerHW::BufferHW &LayerHW::getInput() const
{
	return _input;
}

const LayerHW::BufferHW &LayerHW::getOutput() const
{
	return _output;
}

void LayerHW::_update()
{
	cl::work_range range({getSize()});
	getKernel("update_uniform")->evaluate(range, getSize(), _input.getBuffer(), _output.getBuffer());
}
