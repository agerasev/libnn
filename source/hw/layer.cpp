#include <nn/hw/layer.hpp>

LayerHW::LayerHW(ID id, int size, cl_context context, cl::map<cl::kernel *> *kernels)
	: Layer(id, size), _kernels(kernels)
{
	_input_buffer = new ::cl::buffer_object(context, size*sizeof(cl_float));
	_output_buffer = new ::cl::buffer_object(context, size*sizeof(cl_float));
}

LayerHW::~LayerHW()
{
	delete _input_buffer;
	delete _output_buffer;
}

void LayerHW::_bindQueue(cl_command_queue queue)
{
	_input_buffer->bind_queue(queue);
	_output_buffer->bind_queue(queue);
}

cl::buffer_object *LayerHW::getInput()
{
	return _input_buffer;
}

cl::buffer_object *LayerHW::getOutput() const
{
	return _output_buffer;
}

void LayerHW::_write(const float *data)
{
	_input_buffer->store_data(data);
}

void LayerHW::_read(float *data) const
{
	_output_buffer->load_data(data);
}

void LayerHW::_clear()
{
	float zero = 0.0;
	_kernels["fill"]->evaluate(cl::work_range({unsigned(getSize())}),getSize(),_input_buffer,zero);
}

void LayerHW::_update()
{
	cl::buffer_object *temp_buffer = _input_buffer;
	_input_buffer = _output_buffer;
	_output_buffer = temp_buffer;
}
