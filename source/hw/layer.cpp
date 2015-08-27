#include <nn/hw/layer.hpp>

LayerHW::LayerHW()
	: LayerHW(getID(), getSize(), this)
{
	
}

LayerHW::LayerHW(ID id, int size, const KitHW *kit)
	: Layer(id, size), _input(size, kit), _output(size, kit), KitHW(kit)
{
	
}

LayerHW::~LayerHW()
{
	
}

LayerHW::BufferHW &LayerHW::getInput()
{
	return _input;
}

LayerHW::BufferHW &LayerHW::getOutput()
{
	return _output;
}

const LayerHW::BufferHW &LayerHW::getInput() const
{
	return _input;
}

const LayerHW::BufferHW &LayerHW::getOutput() const
{
	return _output;
}

void LayerHW::_update()
{
	cl::work_range range({getSize()});
	getKernel("update_uniform")->evaluate(range, getSize(), _input.getBuffer(), _output.getBuffer());
}

void LayerHW::_bindQueue(cl::queue *queue)
{
	_input.bindQueue(queue);
	_output.bindQueue(queue);
}
