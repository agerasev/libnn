#include <nn/layer.hpp>


nn::Layer::Layer(ID id, int size)
	: _id(id), _size(size)
{
	
}
	
nn::Layer::ID nn::Layer::getID()
{
	return _id;
}

int nn::Layer::getSize() const
{
	return _size;
}

bool nn::Layer::isValid() const
{
	return _current_valid;
}

void nn::Layer::setValidity(bool v)
{
	_next_valid = v;
}

void nn::Layer::write(const float *data)
{
	_write(data);
	setValidity(true);
}

void nn::Layer::read(float *data) const
{
	_read(data);
}

void nn::Layer::clear()
{
	_clear();
	setValidity(false);
}

void nn::Layer::update()
{
	_update();
	_current_valid = _next_valid;
	_next_valid = false;
}
