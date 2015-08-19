#pragma once

#include <nn/conn.hpp>

class Conn_BP : public virtual Conn
{
protected:
	int _bp_count;
	
public:
	Conn_BP() : Conn(0,0,0) {}
	virtual ~Conn_BP() = default;
	
	virtual void addGrad() = 0;
	virtual void clearGrad() = 0;
	
	virtual void backprop(Layer *to, const Layer *from) = 0;
};
