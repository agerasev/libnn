#pragma once

#include <nn/connection.hpp>
#include <nn/software/layer.hpp>

namespace nn
{
namespace sw
{
class Connection : public nn::Connection
{
private:
	float *_weight_buffer;
	float *_bias_buffer;
	int _weight_buffer_size;
	int _bias_buffer_size;
	
protected:
	Connection(ID id, int input_size, int output_size, int weight_buffer_size, int bias_buffer_size);
	
public:
	Connection(ID id, int input_size, int output_size);
	virtual ~Connection();
	
	float *getWeightBuffer();
	const float *getWeightBuffer() const;
	float *getBiasBuffer();
	const float *getBiasBuffer() const;
	
	virtual void _feedforward(const nn::Layer *from, nn::Layer *to) const override;
	
	virtual void read_weight(float *data) const override;
	virtual void read_bias(float *data) const override;
	virtual void write_weight(const float *data) override;
	virtual void write_bias(const float *data) override;
};
}
}
