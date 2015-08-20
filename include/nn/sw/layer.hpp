#pragma once

#include <nn/layer.hpp>
#include <nn/sw/buffer.hpp>

class LayerSW : public virtual Layer
{
public:
	class BufferSW : public virtual ::BufferSW, public virtual Layer::Buffer
	{
	protected:
		BufferSW() : BufferSW(getSize()) {}
	public:
		BufferSW(int size) : ::Buffer(size) {}
		virtual ~BufferSW() = default;
		
		virtual void write(const float *data) override;
		virtual void clear() override;
	};
	
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
