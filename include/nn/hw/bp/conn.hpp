#pragma once

#include <nn/bp/conn.hpp>
#include <nn/hw/conn.hpp>

class ConnHW_BP : public virtual ConnHW, public virtual Conn_BP
{
private:
	ConnHW::BufferHW _weight_grad;
	ConnHW::BufferHW _bias_grad;
	
	// TODO: add weight_size and bias_size to ctor
protected:
	ConnHW_BP();
public:
	ConnHW_BP(ID id, int input_size, int output_size, const KitHW *kit);
	virtual ~ConnHW_BP() = default;
	
	virtual ConnHW::BufferHW &getWeightGrad() override;
	virtual ConnHW::BufferHW &getBiasGrad() override;
	virtual const ConnHW::BufferHW &getWeightGrad() const override;
	virtual const ConnHW::BufferHW &getBiasGrad() const override;
	
protected:
	virtual void _commitGrad(float delta) override;
	virtual void _backprop(const Layer *to, const Layer_BP *from) override;
	virtual void _backprop(Layer_BP *to, const Layer_BP *from) override;
};
