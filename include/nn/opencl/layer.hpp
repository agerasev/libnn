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
	Layer(ID id, int size, cl_context context, ::cl::kernel *kernel_fill);
	virtual ~Layer();
	
	void bindQueue(cl_command_queue queue);
	cl_command_queue getQueue() const;
	
	::cl::buffer_object *getInputBuffer();
	::cl::buffer_object *getOutputBuffer() const;
	
private:
	virtual void _write(const float *data) override;
	virtual void _read(float *data) const override;
	virtual void _clear() override;
	virtual void _update() override;
};
}
}
