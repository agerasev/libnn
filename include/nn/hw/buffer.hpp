#pragma once

#include <nn/buffer.hpp>
#include <nn/hw/queueable.hpp>
#include <nn/hw/kernelmap.h>

#include <cl/context.hpp>
#include <cl/buffer_object.hpp>
#include <cl/map.hpp>
#include <cl/kernel.hpp>

class BufferHW : 
        public virtual Buffer, 
        public virtual QueueableHW,
        public virtual KernelMapHW
{
private:
	cl::buffer_object _buffer;
	
protected:
	BufferHW(cl::context context);
public:
	BufferHW(int size, cl::context context, const cl::map<cl::kernel *> *kernels);
	virtual ~BufferHW();
	
	virtual void read(float *data) const override;
	virtual void write(const float *data) override;
	virtual void clear() override;
	
	cl::buffer_object *getBuffer();
	const cl::buffer_object *getBuffer() const;
	
protected:
	virtual void _bindQueue(cl_command_queue queue) override;
};
