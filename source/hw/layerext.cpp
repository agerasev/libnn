#include <nn/hw/layerext.hpp>
#include <nn/exception.hpp>

LayerExtHW<LayerFunc::UNIFORM>::LayerExtHW(ID id, int size, const KitHW *kit)
    : Layer(id, size), KitHW(kit)
{
	
}

LayerExtHW<LayerFunc::SIGMOID>::LayerExtHW(ID id, int size, const KitHW *kit)
    : Layer(id, size), KitHW(kit)
{
	
}

void LayerExtHW<LayerFunc::SIGMOID>::_update()
{
	cl::work_range range({getSize()});
	getKernel("update_sigmoid")->evaluate(range, getSize(), _input.getBuffer(), _output.getBuffer());
}

