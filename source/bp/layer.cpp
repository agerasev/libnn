#include <nn/bp/layer.hpp>

void Layer_BP::updateError()
{
	if(getOutputError().isValid())
	{
		_updateError();
		getInputError().validate(true);
		getOutputError().validate(false);
	}
	else
	{
		getInputError().validate(false);
	}
}

void Layer_BP::setDesiredOutput(float *result)
{
	_setDesiredOutput(result);
	getOutputError().validate(true);
}
