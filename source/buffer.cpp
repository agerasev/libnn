#include <nn/buffer.hpp>

Buffer::Buffer(int size)
{
	_size = size;
}

int Buffer::getSize() const
{
	return _size;
}
