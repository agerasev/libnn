#include <nn/sw/layerext.hpp>

#include <cmath>
#include <nn/exception.hpp>

LayerExtSW<EXT_NONE>::LayerExtSW(ID id, int size)
    : Layer(id, size), LayerSW(id, size)
{
	
}

LayerExtSW<EXT_SIGMOID>::LayerExtSW(ID id, int size)
    : Layer(id, size), LayerSW(id, size)
{
	
}

float LayerExtSW<EXT_SIGMOID>::_sigma(float a)
{
	return 1.0f/(1.0f + exp(-a));
}

void LayerExtSW<EXT_SIGMOID>::_update()
{
	if(getInput().getSize() != getOutput().getSize())
		throw Exception("input and output buffers sizes doesn't match");
	
	int size = getInput().getSize();
	float *input = getInput().getData(), *output = getOutput().getData();
	for(int i = 0; i < size; ++i)
		output[i] = _sigma(input[i]);
}
