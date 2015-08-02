#pragma once

#include <cl/kernel.hpp>
#include <cl/buffer_object.hpp>

#include <nn/connection.hpp>

#include <nn/opencl/layer.hpp>

namespace nn
{
namespace cl
{
class Connection : public nn::Connection
{
private:
	::cl::kernel *_kernel;
	::cl::buffer_object *_weight_buffer;
	::cl::buffer_object *_bias_buffer;
	cl_command_queue _queue;
	
public:
	Connection(ID id, int input_size, int output_size, ::cl::kernel *kernel, cl_context context, 
	           int weight_buffer_size = -1, int bias_buffer_size = -1)
	  : nn::Connection(id,input_size, output_size), _kernel(kernel)
	{
		if(weight_buffer_size < 0)
			weight_buffer_size = input_size*output_size;
		if(bias_buffer_size < 0)
			bias_buffer_size = output_size;
		_weight_buffer = new ::cl::buffer_object(context, weight_buffer_size*sizeof(cl_float));
		_bias_buffer = new ::cl::buffer_object(context, bias_buffer_size*sizeof(cl_float));
	}
	
	virtual ~Connection()
	{
		delete _weight_buffer;
		delete _bias_buffer;
	}
	
	void bindQueue(cl_command_queue queue)
	{
		_queue = queue;
		_weight_buffer->bind_queue(queue);
		_bias_buffer->bind_queue(queue);
	}
	
	cl_command_queue getQueue() const
	{
		return _queue;
	}
	
	::cl::buffer_object *getWeightBuffer()
	{
		return _weight_buffer;
	}
	
	const ::cl::buffer_object *getWeightBuffer() const
	{
		return _weight_buffer;
	}
	
	::cl::buffer_object *getBiasBuffer()
	{
		return _bias_buffer;
	}
	
	const ::cl::buffer_object *getBiasBuffer() const
	{
		return _bias_buffer;
	}
	
	virtual void _feedforward(const nn::Layer *from, nn::Layer *to) const override
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
	
	virtual void read_weight(float *data) const override
	{
		_weight_buffer->load_data(data);
	}
	
	virtual void read_bias(float *data) const override
	{
		_bias_buffer->load_data(data);
	}
	
	virtual void write_weight(const float *data) override
	{
		_weight_buffer->store_data(data);
	}

	virtual void write_bias(const float *data) override
	{
		_bias_buffer->store_data(data);
	}
};
}
}
