#pragma once

#include <nn/layer.hpp>
#include <nn/hw/buffer.hpp>
#include <nn/hw/queueable.hpp>
#include <nn/hw/kernelmap.h>

#include <cl/context.hpp>
#include <cl/map.hpp>
#include <cl/kernel.hpp>
#include <cl/queue.hpp>

class LayerHW : 
        public virtual Layer, 
        public virtual QueueableHW, 
        public virtual KernelMapHW
{
public:
	class BufferHW : 
	        public virtual ::BufferHW, 
	        public virtual Layer::Buffer
	{
	protected:
		BufferHW() : BufferHW(getSize()) {}
	public:
		BufferHW(int size) : ::Buffer(size) {}
		virtual ~BufferHW() = default;
		
		virtual void write(const float *data) override;
		virtual void clear() override;
	};
	
private:
	BufferHW _input;
	BufferHW _output;
	
protected:
	LayerHW(cl::context context, const cl::map<cl::kernel *> *kernels);
public:
	LayerHW(ID id, int size, cl::context context, const cl::map<cl::kernel *> *kernels);
	virtual ~LayerHW();
	
	virtual BufferHW &getInput() override;
	virtual BufferHW &getOutput() override;
	virtual const BufferHW &getInput() const override;
	virtual const BufferHW &getOutput() const override;
	
protected:
	virtual void _update() override;
};
