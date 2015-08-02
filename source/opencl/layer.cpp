#include <nn/opencl/layer.hpp>

nn::cl::Layer::Layer(ID id, int size, cl_context context, ::cl::kernel *kernel_fill)
	: nn::Layer(id, size)
{
	_input_buffer = new ::cl::buffer_object(context, size*sizeof(cl_float));
	_output_buffer = new ::cl::buffer_object(context, size*sizeof(cl_float));
	_kernel_fill = kernel_fill;
}

nn::cl::Layer::~Layer()
{
	delete _input_buffer;
	delete _output_buffer;
}

void nn::cl::Layer::bindQueue(cl_command_queue queue)
{
	_queue = queue;
	_input_buffer->bind_queue(queue);
	_output_buffer->bind_queue(queue);
}

cl_command_queue nn::cl::Layer::getQueue() const
{
	return _queue;
}

cl::buffer_object *nn::cl::Layer::getInputBuffer()
{
	return _input_buffer;
}

cl::buffer_object *nn::cl::Layer::getOutputBuffer() const
{
	return _output_buffer;
}

void nn::cl::Layer::_write(const float *data)
{
	_input_buffer->store_data(data);
}

void nn::cl::Layer::_read(float *data) const
{
	_output_buffer->load_data(data);
}

void nn::cl::Layer::_clear()
{
	float zero = 0.0;
	_kernel_fill->evaluate(::cl::work_range({unsigned(getSize())}),getSize(),_input_buffer,zero);
}

void nn::cl::Layer::_update()
{
	::cl::buffer_object *temp_buffer = _input_buffer;
	_input_buffer = _output_buffer;
	_output_buffer = temp_buffer;
}
