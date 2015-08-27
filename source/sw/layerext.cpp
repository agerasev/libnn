#include <nn/sw/layerext.hpp>

#include <cmath>
#include <nn/exception.hpp>

LayerExtSW<LayerFunc::UNIFORM>::LayerExtSW(ID id, int size)
    : Layer(id, size)
{
	
}

LayerExtSW<LayerFunc::SIGMOID>::LayerExtSW(ID id, int size)
    : Layer(id, size)
{
	
}

float LayerExtSW<LayerFunc::SIGMOID>::_sigma(float a)
{
	return 1.0f/(1.0f + exp(-a));
}

void LayerExtSW<LayerFunc::SIGMOID>::_update()
{
	if(getInput().getSize() != getOutput().getSize())
		throw Exception("input and output buffers sizes doesn't match");
	
	int size = getInput().getSize();
	float const *input = getInput().getData();
	float *output = getOutput().getData();
	for(int i = 0; i < size; ++i)
		output[i] = _sigma(input[i]);
}
