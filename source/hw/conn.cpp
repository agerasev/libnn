#include <nn/hw/conn.hpp>

ConnHW::ConnHW(
    ID id, int input_size, int output_size, 
    cl_context context, cl::map<cl::kernel *> &kernels,
    int weight_size, int bias_size
    )
	: Conn(id, input_size, output_size), _kernels(kernels)
{
	_weight_buffer = new cl::buffer_object(
	      context, weight_size*sizeof(cl_float)
	      );
	_bias_buffer = new cl::buffer_object(
	      context, bias_size*sizeof(cl_float)
	      );
}

ConnHW::ConnHW(
    ID id, int input_size, int output_size, 
    cl_context context, cl::map<cl::kernel *> &kernels
    )
  : ConnHW(id, input_size, output_size, context, kernels, input_size*output_size, output_size)
{
	
}

ConnHW::~ConnHW()
{
	delete _weight_buffer;
	delete _bias_buffer;
}

void ConnHW::_bindQueue(cl_command_queue queue)
{
	_weight_buffer->bind_queue(queue);
	_bias_buffer->bind_queue(queue);
}

cl::buffer_object *ConnHW::getWeight()
{
	return _weight_buffer;
}

const cl::buffer_object *ConnHW::getWeight() const
{
	return _weight_buffer;
}

cl::buffer_object *ConnHW::getBias()
{
	return _bias_buffer;
}

const cl::buffer_object *ConnHW::getBias() const
{
	return _bias_buffer;
}

void ConnHW::_transmit(const Layer *from, Layer *to) const
{
	const LayerHW *input = dynamic_cast<const LayerHW *>(from);
	if(input == nullptr)
		throw Exception("input layer is not derived from LayerHW");
	
	LayerHW *output = dynamic_cast<LayerHW *>(to);
	if(output == nullptr)
		throw Exception("output layer is not derived from LayerHW");
	
	::cl::work_range range({unsigned(getOutputSize())});
	_kernels["transmit"]->evaluate(
				range, getInputSize(), getOutputSize(),
				input->getOutput(), output->getInput(), 
				_weight_buffer, _bias_buffer
				);
}

void ConnHW::readWeight(float *data) const
{
	_weight_buffer->load_data(data);
}

void ConnHW::readBias(float *data) const
{
	_bias_buffer->load_data(data);
}

void ConnHW::writeWeight(const float *data)
{
	_weight_buffer->store_data(data);
}

void ConnHW::writeBias(const float *data)
{
	_bias_buffer->store_data(data);
}
