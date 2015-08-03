#include <nn/layer.hpp>


Layer::Layer(ID id, int size)
	: _id(id), _size(size)
{
	
}
	
Layer::ID Layer::getID()
{
	return _id;
}

int Layer::getSize() const
{
	return _size;
}

bool Layer::isZero() const
{
	return _zero_out;
}

void Layer::setZero(bool v)
{
	_zero_in = _zero_in && v;
}

void Layer::write(const float *data)
{
	_write(data);
	_zero_in = false;
}

void Layer::read(float *data) const
{
	_read(data);
}

void Layer::clear()
{
	_clear();
	_zero_in = true;
}

void Layer::update()
{
	_update();
	_zero_out = _zero_in;
	_zero_in = true;
}
