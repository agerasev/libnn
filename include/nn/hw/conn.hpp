#pragma once

#include <nn/conn.hpp>
#include <nn/hw/layer.hpp>
#include <nn/hw/buffer.hpp>
#include <nn/hw/kit.hpp>

class ConnHW : 
        public virtual Conn,
        public virtual KitHW
{
public:
	class BufferHW : 
	        public virtual ::BufferHW, 
	        public virtual Buffer
	{
	protected:
		BufferHW() : BufferHW(getSize(), this) {}
	public:
		BufferHW(int size, const KitHW *kit) : ::Buffer(size), KitHW(kit) {}
		virtual ~BufferHW() = default;
		
		virtual void randomize(float range = 1.0f) override;
	};
		
private:
	BufferHW _weight;
	BufferHW _bias;
	
protected:
	ConnHW(ID id, int input_size, int output_size, int weight_size, int bias_size, const KitHW *kit);
	ConnHW();
public:
	ConnHW(ID id, int input_size, int output_size, const KitHW *kit);
	virtual ~ConnHW();
	
	virtual BufferHW &getWeight() override;
	virtual BufferHW &getBias() override;
	virtual const BufferHW &getWeight() const override;
	virtual const BufferHW &getBias() const override;
	
protected:
	virtual void _transmit(const Layer *from, Layer *to) const override;
	virtual void _bindQueue(cl::queue *queue) override;
};
