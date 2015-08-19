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
