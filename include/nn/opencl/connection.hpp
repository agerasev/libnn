#pragma once

#include <cl/kernel.hpp>
#include <cl/buffer_object.hpp>

#include <nn/exception.hpp>
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
	Connection(
	    ID id, int input_size, int output_size, ::cl::kernel *kernel, 
	    cl_context context, int weight_buffer_size = -1, int bias_buffer_size = -1
	    );
	virtual ~Connection();
	
	void bindQueue(cl_command_queue queue);
	cl_command_queue getQueue() const;
	
	::cl::buffer_object *getWeightBuffer();
	const ::cl::buffer_object *getWeightBuffer() const;
	::cl::buffer_object *getBiasBuffer();
	const ::cl::buffer_object *getBiasBuffer() const;
	
	virtual void _feedforward(const nn::Layer *from, nn::Layer *to) const override;
	
	virtual void read_weight(float *data) const override;
	virtual void read_bias(float *data) const override;
	virtual void write_weight(const float *data) override;
	virtual void write_bias(const float *data) override;
};
}
}
