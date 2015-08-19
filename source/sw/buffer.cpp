#include <nn/sw/buffer.hpp>

#include <typeinfo>
#include <nn/exception.hpp>

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
	setZero(false);
	validate(true);
}

void BufferSW::clear()
{
	const int size = getSize();
	for(int i = 0; i < size; ++i)
	{
		_data[i] = 0.0f;
	}
	setZero(true);
}

void BufferSW::copy(const Buffer &buffer)
{
	try
	{
		const BufferSW &sw_buffer = dynamic_cast<const BufferSW &>(buffer);
		copy(sw_buffer);
	}
	catch(std::bad_cast)
	{
		throw Exception("buffer is not derived by BufferSW");
	}
}

void BufferSW::copy(const BufferSW &buffer)
{
	if(buffer.getSize() != getSize())
		throw Exception("buffers sizes doesn't match");
	
	const float *data = buffer.getData();
	write(data);
}

float *BufferSW::getData()
{
	return _data;
}

const float *BufferSW::getData() const
{
	return _data;
}
