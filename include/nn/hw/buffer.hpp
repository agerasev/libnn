#pragma once

#include <nn/buffer.hpp>
#include <nn/hw/kit.hpp>

#include <cl/context.hpp>
#include <cl/buffer_object.hpp>
#include <cl/map.hpp>
#include <cl/kernel.hpp>

class BufferHW : 
        public virtual Buffer, 
        public virtual KitHW
{
private:
	cl::buffer_object _buffer;
	
protected:
	BufferHW();
public:
	BufferHW(int size, const KitHW *kit);
	virtual ~BufferHW();
	
	virtual void read(float *data) const override;
	virtual void write(const float *data) override;
	virtual void clear() override;
	
	cl::buffer_object *getBuffer();
	const cl::buffer_object *getBuffer() const;
	
protected:
	virtual void _bindQueue(cl::queue *queue) override;
};
