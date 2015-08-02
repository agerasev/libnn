#pragma once

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
	Connection(ID id, int input_size, int output_size);
	virtual ~Connection() = default;
	
	ID getID() const;
	int getInputSize() const;
	int getOutputSize() const;
	
	virtual void read_weight(float *data) const = 0;
	virtual void read_bias(float *data) const = 0;
	virtual void write_weight(const float *data) = 0;
	virtual void write_bias(const float *data) = 0;
	
	virtual void feedforward(const Layer *from, Layer *to) const;
};
}
