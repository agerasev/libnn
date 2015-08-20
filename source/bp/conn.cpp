#include <nn/bp/conn.hpp>
#include <nn/bp/layer.hpp>
#include <nn/exception.hpp>

Conn_BP::getBPCount() const
{
	return _bp_count;
}

void Conn_BP::incBPCount()
{
	_bp_count += 1;
}

void Conn_BP::clearBPCount()
{
	_bp_count = 0;
}

void Conn_BP::commitGrad(float delta)
{
	_commitGrad(delta);
	clearBPCount();
}

void Conn_BP::backprop(Layer *to, const Layer *from)
{
	if(from->getSize() != getInputSize())
		throw Exception("input layer and connection sizes do not match");
	
	if(to->getSize() != getOutputSize())
		throw Exception("output layer and connection sizes do not match");
	
	Layer_BP *to_bp = dynamic_cast<Layer_BP *>(to);
	const Layer_BP *from_bp = dynamic_cast<const Layer_BP *>(from);
	if(from_bp == nullptr)
		throw Exception("input layer is not derived from Layer_BP");
	
	if(from_bp->getOutputError().isValid())
	{
		if(to_bp != nullptr)
		{
			if(!to_bp->getInputError().isValid())
			{
				if(!to_bp->getInputError().isZero())
					to_bp->getInputError().clear();
				to_bp->getInputError().validate(true);
			}
			_backprop(to_bp, from_bp);
			to_bp->getInputError().setZero(false);
		}
		else
		{
			_backprop(to, from_bp);
		}
		incBPCount();
	}
}
