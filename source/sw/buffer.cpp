#include <nn/sw/buffer.hpp>

BufferSW::BufferSW()
    : BufferSW(getSize())
{
	
}

BufferSW::BufferSW(int size)
    : Buffer(size)
{
	_data = new float[size];
}

BufferSW::~BufferSW()
{
	delete[] _data;
}

void BufferSW::read(float *data) const
{
	const int size = getSize();
	for(int i = 0; i < size; ++i)
	{
		data[i] = _data[i];
	}
}

void BufferSW::write(const float *data)
{
	const int size = getSize();
	for(int i = 0; i < size; ++i)
	{
		_data[i] = data[i];
	}
}

void BufferSW::clear()
{
	const int size = getSize();
	for(int i = 0; i < size; ++i)
	{
		_data[i] = 0.0f;
	}
}

float *BufferSW::getData()
{
	return _data;
}

const float *BufferSW::getData() const
{
	return _data;
}
