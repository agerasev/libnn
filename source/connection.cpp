#include <nn/exception.hpp>
#include <nn/connection.hpp>

nn::Connection::Connection(ID id, int input_size, int output_size)
	: _id(id), _in_size(input_size), _out_size(output_size)
{
	
}

nn::Connection::ID nn::Connection::getID() const
{
	return _id;
}

int nn::Connection::getInputSize() const
{
	return _in_size;
}

int nn::Connection::getOutputSize() const
{
	return _out_size;
}

void nn::Connection::feedforward(const Layer *from, Layer *to) const
{
	if(from->getSize() != getInputSize())
		throw Exception("input buffer and connection sizes do not match");
	
	if(to->getSize() != getOutputSize())
		throw Exception("output buffer and connection sizes do not match");
	
	if(from->isValid())
	{
		_feedforward(from, to);
		to->setValidity(true);
	}
}

