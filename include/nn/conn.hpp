#pragma once

#include "layer.hpp"

class Conn
{
public:
	typedef unsigned ID;
	
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
	
	virtual void readWeight(float *data) const = 0;
	virtual void readBias(float *data) const = 0;
	virtual void writeWeight(const float *data) = 0;
	virtual void writeBias(const float *data) = 0;
	
	// virtual void randomizeWeight() = 0;
	// virtual void randomizeBias() = 0;
	
	virtual void transmit(const Layer *from, Layer *to) const;
};
