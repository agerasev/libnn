#pragma once

#include <nn/buffer.hpp>

class BufferSW : public virtual Buffer
{
private:
	float *_data;
	
protected:
	BufferSW();
public:
	BufferSW(int size);
	virtual ~BufferSW();
	
	virtual void read(float *data) const override;
	virtual void write(const float *data) override;
	virtual void clear() override;
	
	virtual void copy(const Buffer &buffer) override;
	virtual void copy(const BufferSW &buffer);
	
	float *getData();
	const float *getData() const;
};
