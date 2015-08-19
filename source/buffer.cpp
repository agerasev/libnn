#include <nn/buffer.hpp>

Buffer::Buffer(int size)
{
	_size = size;
}

int Buffer::getSize() const
{
	return _size;
}

void Buffer::setZero(bool z)
{
	_zero = z;
}

bool Buffer::isZero() const
{
	return _zero;
}

void Buffer::validate(bool v)
{
	_validity = v;
}

bool Buffer::isValid() const
{
	return _validity;
}
