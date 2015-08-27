#pragma once

#include <nn/layer.hpp>
#include <nn/hw/buffer.hpp>
#include <nn/hw/kit.hpp>

#include <cl/context.hpp>
#include <cl/map.hpp>
#include <cl/kernel.hpp>
#include <cl/queue.hpp>

class LayerHW : 
        public virtual Layer, 
        public virtual KitHW
{
public:
	class BufferHW : 
	        public virtual ::BufferHW, 
	        public virtual Layer::Buffer
	{
	protected:
		BufferHW() : BufferHW(getSize(), this) {}
	public:
		BufferHW(int size, const KitHW *kit) : ::Buffer(size), KitHW(kit) {}
		virtual ~BufferHW() = default;
		
		virtual void write(const float *data) override;
		virtual void clear() override;
	};
	
private:
	BufferHW _input;
	BufferHW _output;
	
protected:
	LayerHW();
public:
	LayerHW(ID id, int size, const KitHW *kit);
	virtual ~LayerHW();
	
	virtual BufferHW &getInput() override;
	virtual BufferHW &getOutput() override;
	virtual const BufferHW &getInput() const override;
	virtual const BufferHW &getOutput() const override;
	
protected:
	virtual void _update() override;
	virtual void _bindQueue(cl::queue *queue) override;
};
