#include <nn/exception.hpp>
#include <nn/conn.hpp>

Conn::Conn(ID id, int input_size, int output_size)
	: _id(id), _in_size(input_size), _out_size(output_size)
{
	
}

Conn::ID Conn::getID() const
{
	return _id;
}

int Conn::getInputSize() const
{
	return _in_size;
}

int Conn::getOutputSize() const
{
	return _out_size;
}

void Conn::transmit(const Layer *from, Layer *to) const
{
	if(from->getOutput().getSize() != getInputSize())
		throw Exception("input buffer and connection sizes do not match");
	
	if(to->getInput().getSize() != getOutputSize())
		throw Exception("output buffer and connection sizes do not match");
	
	if(from->getOutput().isValid())
	{
		if(!to->getInput().isValid())
		{
			if(!to->getInput().isZero())
				to->getInput().clear();
			to->getInput().validate(true);
		}
		_transmit(from, to);
		to->getInput().setZero(false);
	}
}

