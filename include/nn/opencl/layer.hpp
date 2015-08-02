#pragma once

#include <cl/buffer_object.hpp>
#include <cl/kernel.hpp>

#include <nn/layer.hpp>

namespace nn
{
namespace cl
{
class Layer : public nn::Layer
{
private:
	::cl::buffer_object *_input_buffer, *_output_buffer;
	cl_command_queue _queue;
	::cl::kernel *_kernel_fill;
	
public:
	Layer(ID id, int size, cl_context context, ::cl::kernel *kernel_fill)
	  : nn::Layer(id, size)
	{
		_input_buffer = new ::cl::buffer_object(context, size*sizeof(cl_float));
	  _output_buffer = new ::cl::buffer_object(context, size*sizeof(cl_float));
		_kernel_fill = kernel_fill;
	}
	
	virtual ~Layer()
	{
		delete _input_buffer;
		delete _output_buffer;
	}
	
	void bindQueue(cl_command_queue queue)
	{
		_queue = queue;
		_input_buffer->bind_queue(queue);
		_output_buffer->bind_queue(queue);
	}
	
	cl_command_queue getQueue() const
	{
		return _queue;
	}
	
	::cl::buffer_object *getInputBuffer()
	{
		return _input_buffer;
	}
	
	::cl::buffer_object *getOutputBuffer() const
	{
		return _output_buffer;
	}
	
protected:
	virtual void _write(const float *data) override
	{
		_input_buffer->store_data(data);
	}
	
	virtual void _read(float *data) const override
	{
		_output_buffer->load_data(data);
	}
	
	virtual void _clear() override
	{
		float zero = 0.0;
		_kernel_fill->evaluate(::cl::work_range({unsigned(getSize())}),getSize(),_input_buffer,zero);
	}
	
	virtual void _update() override
	{
		::cl::buffer_object *temp_buffer = _input_buffer;
		_input_buffer = _output_buffer;
		_output_buffer = temp_buffer;
	}
};
}
}
