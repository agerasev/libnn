#include <nn/bp/layer.hpp>

void Layer_BP::updateError()
{
	if(getInputError().isValid())
	{
		_updateError();
		getOutputError().validate(true);
		getInputError().validate(false);
	}
	else
	{
		getOutputError().validate(false);
	}
}
