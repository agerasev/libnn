#pragma once

#include <nn/conn.hpp>
#include <nn/sw/layer.hpp>

class ConnSW : public virtual Conn
{
private:
	float *_weight_buffer;
	float *_bias_buffer;
	int _weight_size;
	int _bias_size;
	
protected:
	ConnSW(ID id, int input_size, int output_size, int weight_size, int bias_size);
	
public:
	ConnSW(ID id, int input_size, int output_size);
	virtual ~ConnSW();
	
	float *getWeight();
	const float *getWeight() const;
	float *getBias();
	const float *getBias() const;
	
	virtual void _transmit(const Layer *from, Layer *to) const override;
	
	virtual void readWeight(float *data) const override;
	virtual void readBias(float *data) const override;
	virtual void writeWeight(const float *data) override;
	virtual void writeBias(const float *data) override;
};
