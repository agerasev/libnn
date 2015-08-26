#include <nn/hw/buffer.hpp>

BufferHW::BufferHW()
    : BufferHW(getSize())
{
	
}

BufferHW::BufferHW(cl_context context, int size)
    : Buffer(size), _buffer(context, size)
{
	
}

BufferHW::~BufferHW()
{
	
}

void BufferHW::read(float *data) const
{
	_buffer.load_data(data);
}

void BufferHW::write(const float *data)
{
	_buffer.store_data(data);
}

void BufferHW::clear()
{
	
}

float *BufferHW::getData()
{
	return _data;
}

const float *BufferHW::getData() const
{
	return _data;
}

