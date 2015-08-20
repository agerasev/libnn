#pragma once

#include <nn/bp/conn.hpp>
#include <nn/sw/conn.hpp>

class ConnSW_BP : public virtual ConnSW, public virtual Conn_BP
{
private:
	ConnSW::BufferSW _weight_grad;
	ConnSW::BufferSW _bias_grad;
	
	// TODO: add weight_size and bias_size to ctor
protected:
	ConnSW_BP() : ConnSW_BP(getID(), getInputSize(), getOutputSize()) {}
public:
	ConnSW_BP(ID id, int input_size, int output_size);
	virtual ~ConnSW_BP() = default;
	
	virtual ConnSW::BufferSW &getWeightGrad() override;
	virtual ConnSW::BufferSW &getBiasGrad() override;
	virtual const ConnSW::BufferSW &getWeightGrad() const override;
	virtual const ConnSW::BufferSW &getBiasGrad() const override;
	
protected:
	virtual void _commitGrad(float delta) override;
	virtual void _backprop(const Layer *to, const Layer_BP *from) override;
	virtual void _backprop(Layer_BP *to, const Layer_BP *from) override;
};
