#pragma once

#include <nn/conn.hpp>
#include <nn/sw/layer.hpp>
#include <nn/sw/buffer.hpp>

class ConnSW : public virtual Conn
{
public:
	class BufferSW : public virtual ::BufferSW, public virtual Buffer
	{
	protected:
		BufferSW() : BufferSW(getSize()) {}
	public:
		BufferSW(int size) : ::Buffer(size) {}
		virtual ~BufferSW() = default;
		
		virtual void randomize(float range = 1.0f) override;
	};
		
private:
	BufferSW _weight;
	BufferSW _bias;
	
protected:
	ConnSW(ID id, int input_size, int output_size, int weight_size, int bias_size);
	ConnSW();
public:
	ConnSW(ID id, int input_size, int output_size);
	virtual ~ConnSW() = default;
	
	virtual BufferSW &getWeight() override;
	virtual BufferSW &getBias() override;
	virtual const BufferSW &getWeight() const override;
	virtual const BufferSW &getBias() const override;
	
protected:
	virtual void _transmit(const Layer *from, Layer *to) const override;
};
