#include <nn/sw/layer.hpp>

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

BufferSW &LayerSW::getInput()
{
	return _input;
}

BufferSW &LayerSW::getOutput()
{
	return _output;
}

const BufferSW &LayerSW::getInput() const
{
	return _input;
}

const BufferSW &LayerSW::getOutput() const
{
	return _output;
}

void LayerSW::_update()
{
	_output.copy(_input);
}
