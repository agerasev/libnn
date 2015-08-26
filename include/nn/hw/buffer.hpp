#pragma once

#include <nn/buffer.hpp>
#include <nn/hw/queueable.hpp>

#include <cl/buffer_object.hpp>
#include <cl/map.hpp>
#include <cl/kernel.hpp>

class BufferHW : public virtual Buffer, public virtual QueueableHW
{
private:
	cl::map<cl::kernel *> *_kernels;
	cl::buffer_object _buffer;
	
protected:
	BufferHW();
public:
	BufferHW(int size, cl_context context, cl::map<cl::kernel *> *kernels);
	virtual ~BufferHW();
	
	virtual void read(float *data) const override;
	virtual void write(const float *data) override;
	virtual void clear() override;
	
	cl_mem getCLMem() const;
};
