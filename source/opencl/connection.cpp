#include <nn/opencl/connection.hpp>

nn::cl::Connection::Connection(
    ID id, int input_size, int output_size, 
    ::cl::kernel *kernel, cl_context context, 
    int weight_buffer_size, int bias_buffer_size
    )
	: nn::Connection(id,input_size, output_size), _kernel(kernel)
{
	if(weight_buffer_size < 0)
		weight_buffer_size = input_size*output_size;
	if(bias_buffer_size < 0)
		bias_buffer_size = output_size;
	
	_weight_buffer = new ::cl::buffer_object(
	      context, weight_buffer_size*sizeof(cl_float)
	      );
	_bias_buffer = new ::cl::buffer_object(
	      context, bias_buffer_size*sizeof(cl_float)
	      );
}

nn::cl::Connection::~Connection()
{
	delete _weight_buffer;
	delete _bias_buffer;
}

void nn::cl::Connection::bindQueue(cl_command_queue queue)
{
	_queue = queue;
	_weight_buffer->bind_queue(queue);
	_bias_buffer->bind_queue(queue);
}

cl_command_queue nn::cl::Connection::getQueue() const
{
	return _queue;
}

cl::buffer_object *nn::cl::Connection::getWeightBuffer()
{
	return _weight_buffer;
}

const cl::buffer_object *nn::cl::Connection::getWeightBuffer() const
{
	return _weight_buffer;
}

cl::buffer_object *nn::cl::Connection::getBiasBuffer()
{
	return _bias_buffer;
}

const cl::buffer_object *nn::cl::Connection::getBiasBuffer() const
{
	return _bias_buffer;
}

void nn::cl::Connection::_feedforward(const nn::Layer *from, nn::Layer *to) const
{
	const nn::cl::Layer *input = dynamic_cast<const nn::cl::Layer *>(from);
	if(input == nullptr)
		throw Exception("input layer is not derived from nn::cl::Layer");
	
	nn::cl::Layer *output = dynamic_cast<nn::cl::Layer *>(to);
	if(output == nullptr)
		throw Exception("output layer is not derived from nn::cl::Layer");
	
	::cl::work_range range({unsigned(getOutputSize())});
	_kernel->evaluate(
				range, getInputSize(), getOutputSize(),
				input->getOutputBuffer(), output->getInputBuffer(), 
				_weight_buffer, _bias_buffer
				);
}

void nn::cl::Connection::read_weight(float *data) const
{
	_weight_buffer->load_data(data);
}

void nn::cl::Connection::read_bias(float *data) const
{
	_bias_buffer->load_data(data);
}

void nn::cl::Connection::write_weight(const float *data)
{
	_weight_buffer->store_data(data);
}

void nn::cl::Connection::write_bias(const float *data)
{
	_bias_buffer->store_data(data);
}
