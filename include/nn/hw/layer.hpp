#pragma once

#include <cl/map.hpp>
#include <cl/kernel.hpp>
#include <cl/buffer_object.hpp>

#include "queueable.hpp"

#include <nn/layer.hpp>

class LayerHW : public virtual Layer, public HWQueueable
{
private:
	cl::buffer_object *_input_buffer, *_output_buffer;
	cl::map<cl::kernel *> &_kernels;
	
protected:
	virtual void _bindQueue(cl_command_queue queue) override;
	
public:
	LayerHW(ID id, int size, cl_context context, cl::map<cl::kernel *> &kernels);
	virtual ~LayerHW();
	
	cl::buffer_object *getInput();
	cl::buffer_object *getOutput() const; 
	
private:
	virtual void _write(const float *data) override;
	virtual void _read(float *data) const override;
	virtual void _clear() override;
	virtual void _update() override;
};
