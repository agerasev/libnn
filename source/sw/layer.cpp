#include <nn/sw/layer.hpp>

#include <nn/exception.hpp>

LayerSW::LayerSW() 
    : LayerSW(getID(), getSize())
{
	
}

LayerSW::LayerSW(ID id, int size)
	: Layer(id, size), _input(size), _output(size)
{
	
}

LayerSW::~LayerSW()
{
	
}

LayerSW::BufferSW &LayerSW::getInput()
{
	return _input;
}

LayerSW::BufferSW &LayerSW::getOutput()
{
	return _output;
}

const LayerSW::BufferSW &LayerSW::getInput() const
{
	return _input;
}

const LayerSW::BufferSW &LayerSW::getOutput() const
{
	return _output;
}

void LayerSW::_update()
{
	if(_input.getSize() != _output.getSize())
		throw Exception("input and outpus sizes doesn't match");
	
	float *output = _output.getData();
	const float *input = _input.getData();
	int size = _input.getSize();
	for(int i = 0; i < size; ++i)
	{
		output[i] = input[i];
	}
}

void LayerSW::BufferSW::write(const float *data)
{
	::BufferSW::write(data);
	setZero(false);
	validate(true);
}

void LayerSW::BufferSW::clear()
{
	::BufferSW::clear();
	setZero(true);
}
