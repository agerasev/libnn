#include <nn/software/sigmoidlayer.hpp>

#include <cmath>

nn::sw::SigmoidLayer::SigmoidLayer(ID id, int size)
    : nn::sw::Layer(id, size)
{
	
}

static float _sigma(float a)
{
	return 1.0f/(1.0f + exp(-a));
}

void nn::sw::SigmoidLayer::_update()
{
	int size = getSize();
	float *input = getInputBuffer(), *output = getOutputBuffer();
	for(int i = 0; i < size; ++i)
		output[i] = _sigma(input[i]);
}
