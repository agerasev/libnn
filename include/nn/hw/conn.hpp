#pragma once

#include <nn/conn.hpp>
#include <nn/hw/layer.hpp>
#include <nn/hw/buffer.hpp>
#include <nn/hw/queueable.hpp>
#include <nn/hw/kernelmap.h>

class ConnHW : 
        public virtual Conn,
        public virtual QueueableHW,
        public virtual KernelMapHW
{
public:
	class BufferHW : 
	        public virtual ::BufferHW, 
	        public virtual Buffer
	{
	protected:
		BufferHW() : BufferHW(getSize()) {}
	public:
		BufferHW(int size) : ::Buffer(size) {}
		virtual ~BufferHW() = default;
		
		virtual void randomize(float range = 1.0f) override;
	};
		
private:
	BufferHW _weight;
	BufferHW _bias;
	
protected:
	ConnHW(ID id, int input_size, int output_size, int weight_size, int bias_size, cl::context context, const cl::map<cl::kernel *> *kernels);
	ConnHW(cl::context context, const cl::map<cl::kernel *> *kernels);
public:
	ConnHW(ID id, int input_size, int output_size, cl::context context, const cl::map<cl::kernel *> *kernels);
	virtual ~ConnHW();
	
	virtual BufferHW &getWeight() override;
	virtual BufferHW &getBias() override;
	virtual const BufferHW &getWeight() const override;
	virtual const BufferHW &getBias() const override;
	
protected:
	virtual void _transmit(const Layer *from, Layer *to) const override;
};
