#pragma once

#include <nn/layer.hpp>
#include <nn/sw/buffer.hpp>

class LayerSW : public virtual Layer
{
private:
	BufferSW _input;
	BufferSW _output;
	
protected:
	LayerSW();
public:
	LayerSW(ID id, int size);
	virtual ~LayerSW();
	
	virtual BufferSW &getInput() override;
	virtual BufferSW &getOutput() override;
	virtual const BufferSW &getInput() const override;
	virtual const BufferSW &getOutput() const override;
	
protected:
	virtual void _update() override;
};
