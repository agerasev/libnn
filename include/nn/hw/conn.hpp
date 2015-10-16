#pragma once

#include <nn/conn.hpp>
#include <nn/hw/layer.hpp>
#include <nn/hw/buffer.hpp>
#include <nn/hw/kit.hpp>

#ifndef NN_NO_OPTIM
#include <vector>
#endif

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
	
#ifndef NN_NO_OPTIM
	static const int REDUCE_FACTOR = 0x8;
	std::vector<BufferHW *> _reduce_buffers;
#endif
	
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
