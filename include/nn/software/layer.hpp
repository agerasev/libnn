#pragma once

#include <nn/layer.hpp>

namespace nn
{
namespace sw
{

class Layer : public nn::Layer
{
private:
	float *_input_buffer;
	float *_output_buffer;
	
public:
	Layer(ID id, int size);
	virtual ~Layer();
	
	float *getInputBuffer();
	const float *getInputBuffer() const;
	float *getOutputBuffer();
	const float *getOutputBuffer() const;
	
protected:
	virtual void _write(const float *data) override;
	virtual void _read(float *data) const override;
	virtual void _clear() override;
	virtual void _update() override;
};

template <class Plugin>
class LayerX : public Layer, public Plugin
{
public:
	template <typename ... Args>
	LayerX(ID id, int size, Args ... args)
	    : Layer(id, size), Plugin(args ...)
	{
		
	}
	
	virtual ~LayerX() = default;
};

}
}
