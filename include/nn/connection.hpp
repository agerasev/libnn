#pragma once

#include "exception.hpp"
#include "layer.hpp"

namespace nn
{
class Connection
{
public:
	typedef unsigned ID;
	
private:
	ID _id;
	int _in_size, _out_size;
	
protected:
	virtual void _feedforward(const Layer *from, Layer *to) const = 0;
	
public:
	Connection(ID id, int input_size, int output_size)
	  : _id(id), _in_size(input_size), _out_size(output_size)
	{
		
	}
	
	virtual ~Connection() = default;
	
	ID getID() const
	{
		return _id;
	}
	
	int getInputSize() const
	{
		return _in_size;
	}
	
	int getOutputSize() const
	{
		return _out_size;
	}
	
	virtual void read_weight(float *data) const = 0;
	virtual void read_bias(float *data) const = 0;
	virtual void write_weight(const float *data) = 0;
	virtual void write_bias(const float *data) = 0;
	
	virtual void feedforward(const Layer *from, Layer *to) const
	{
		if(from->getSize() != getInputSize())
			throw Exception("input buffer and connection sizes do not match");
		
		if(to->getSize() != getOutputSize())
			throw Exception("output buffer and connection sizes do not match");
		
		if(from->isValid())
		{
			_feedforward(from, to);
			to->setValidity(true);
		}
	}
};
}
