#include <nn/hw/layer.hpp>

LayerHW::LayerHW()
	: LayerHW(getID(), getSize(), static_cast<KitHW *>(this))
{
	
}

LayerHW::LayerHW(ID id, int size, const KitHW *kit)
	: Layer(id, size), KitHW(kit), _input(size, kit), _output(size, kit)
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
	cl::work_range range(getSize());
	getKernel("update")->evaluate(range, getSize(), _input.getBuffer(), _output.getBuffer());
}

void LayerHW::_bindQueue(cl::queue *queue)
{
	_input.bindQueue(queue);
	_output.bindQueue(queue);
}

void LayerHW::BufferHW::write(const float *data)
{
	::BufferHW::write(data);
	setZero(false);
	validate(true);
}

void LayerHW::BufferHW::clear()
{
	::BufferHW::clear();
	setZero(true);
}
