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

void Layer::update()
{
	if(getInput().isValid())
	{
		_update();
		getOutput().validate(true);
		getInput().validate(false);
	}
	else
	{
		getOutput().validate(false);
	}
}

void Layer::Buffer::setZero(bool z)
{
	_zero = z;
}

bool Layer::Buffer::isZero() const
{
	return _zero;
}

void Layer::Buffer::validate(bool v)
{
	_validity = v;
}

bool Layer::Buffer::isValid() const
{
	return _validity;
}
