#pragma once

#include <nn/bp/conn.hpp>
#include <nn/sw/conn.hpp>

class ConnSW_BP : public virtual ConnSW, public virtual Conn_BP
{
private:
	float *_weight_grad;
	float *_bias_grad;
	
public:
	ConnSW_BP(ID id, int input_size, int output_size);
	virtual ~ConnSW_BP();
	
	virtual void addGrad() override;
	virtual void clearGrad() override;
	
	virtual void backprop(Layer *to, const Layer *from, float delta) override;
};
