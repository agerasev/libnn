#pragma once

#include <nn/layer.hpp>

class LayerSW : public virtual Layer
{
private:
	float *_input_buffer;
	float *_output_buffer;
	
public:
	LayerSW(ID id, int size);
	virtual ~LayerSW();
	
	float *getInput();
	const float *getInput() const;
	float *getOutput();
	const float *getOutput() const;
	
protected:
	virtual void _write(const float *data) override;
	virtual void _read(float *data) const override;
	virtual void _clear() override;
	virtual void _update() override;
};
