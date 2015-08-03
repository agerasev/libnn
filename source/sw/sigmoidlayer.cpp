#include <nn/sw/sigmoidlayer.hpp>

#include <cmath>

SigmoidLayerSW::SigmoidLayerSW(ID id, int size)
    : Layer(id, size), LayerSW(id, size)
{
	
}

float SigmoidLayerSW::_sigma(float a)
{
	return 1.0f/(1.0f + exp(-a));
}

void SigmoidLayerSW::_update()
{
	int size = getSize();
	float *input = getInput(), *output = getOutput();
	for(int i = 0; i < size; ++i)
		output[i] = _sigma(input[i]);
}
