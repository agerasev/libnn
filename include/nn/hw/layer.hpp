#pragma once

#include "queueable.hpp"

#include <nn/layer.hpp>
#include <nn/hw/buffer.hpp>

class LayerHW : public virtual Layer, public virtual QueueableHW
{
private:
	BufferHW _input, _output;
	cl::map<cl::kernel *> *_kernels;
	
protected:
	virtual void _bindQueue(cl_command_queue queue) override;
	
public:
	LayerHW(ID id, int size, cl_context context, cl::map<cl::kernel *> *kernels);
	virtual ~LayerHW();
	
	cl::buffer_object *getInput();
	cl::buffer_object *getOutput() const; 
	
private:
	virtual void _write(const float *data) override;
	virtual void _read(float *data) const override;
	virtual void _clear() override;
	virtual void _update() override;
};
