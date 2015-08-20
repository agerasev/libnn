#pragma once

#include <nn/conn.hpp>
#include <nn/bp/layer.hpp>

class Conn_BP : public virtual Conn
{
private:
	int _bp_count = 0;
	
protected:
	Conn_BP() : Conn_BP(getID(), getInputSize(), getOutputSize()) {}
public:
	Conn_BP(ID id, int input_size, int output_size) : Conn(id, input_size, output_size) {}
	virtual ~Conn_BP() = default;
	
	int getBPCount() const;
	void incBPCount();
	void clearBPCount();
	
	virtual Conn::Buffer &getWeightGrad() = 0;
	virtual Conn::Buffer &getBiasGrad() = 0;
	virtual const Conn::Buffer &getWeightGrad() const = 0;
	virtual const Conn::Buffer &getBiasGrad() const = 0;
	
	void commitGrad(float delta);
	void backprop(Layer *to, const Layer *from);
	
protected:
	virtual void _commitGrad(float delta) = 0;
	virtual void _backprop(Layer_BP *to, const Layer_BP *from) = 0;
	virtual void _backprop(const Layer *to, const Layer_BP *from) = 0;
};
