#pragma once

#include <nn/layer.hpp>
#include <nn/buffer.hpp>

class Conn
{
public:
	typedef unsigned ID;
	
	class Buffer : public virtual ::Buffer
	{
	private:
		int _size;
		
	protected:
		Buffer() : Buffer(getSize()) {}
	public:
		Buffer(int size) : ::Buffer(size) {}
		virtual ~Buffer() = default;
		
		virtual void randomize(float range = 1.0f) = 0;
	};
	
private:
	ID _id;
	int _in_size, _out_size;
	
protected:
	virtual void _transmit(const Layer *from, Layer *to) const = 0;
	
public:
	Conn(ID id, int input_size, int output_size);
	virtual ~Conn() = default;
	
	ID getID() const;
	int getInputSize() const;
	int getOutputSize() const;
	
public:
	virtual Buffer &getWeight() = 0;
	virtual Buffer &getBias() = 0;
	virtual const Buffer &getWeight() const = 0;
	virtual const Buffer &getBias() const = 0;
	
	virtual void transmit(const Layer *from, Layer *to) const;
};
