#include <nn/hw/bp/conn.hpp>
#include <nn/hw/bp/layer.hpp>
#include <nn/exception.hpp>

#include <nn/hw/utility.hpp>

/*
ConnHW_BP::ConnHW_BP() 
  : ConnHW_BP(getID(), getInputSize(), getOutputSize(), static_cast<const KitHW *>(this))
{
	
}
*/

ConnHW_BP::ConnHW_BP(ID id, int input_size, int output_size, const KitHW *kit)
    : Conn(id, input_size, output_size), KitHW(kit), 
      _weight_grad(input_size*output_size, kit), _bias_grad(output_size, kit)
{
	getWeightGrad().clear();
	getBiasGrad().clear();
}

ConnHW::BufferHW &ConnHW_BP::getWeightGrad()
{
	return _weight_grad;
}

ConnHW::BufferHW &ConnHW_BP::getBiasGrad()
{
	return _bias_grad;
}

const ConnHW::BufferHW &ConnHW_BP::getWeightGrad() const
{
	return _weight_grad;
}

const ConnHW::BufferHW &ConnHW_BP::getBiasGrad() const
{
	return _bias_grad;
}

void ConnHW_BP::_commitGrad(float delta)
{
	const float norm = delta/getBPCount();
	getKernel("commitWeightGrad")->evaluate(
	      cl::work_range(getInputSize(), getOutputSize()), ivec2(getInputSize(), getOutputSize()),
	      norm, getWeightGrad().getBuffer(), getWeight().getBuffer()
	      );
	getKernel("commitBiasGrad")->evaluate(
	      cl::work_range(getOutputSize()), getOutputSize(), norm,
	      getBiasGrad().getBuffer(), getBias().getBuffer()
	      );
	getWeightGrad().clear();
	getBiasGrad().clear();
}

void ConnHW_BP::_backprop(const Layer *to, const Layer_BP *from)
{
	const LayerHW *to_sw = dynamic_cast<const LayerHW *>(to);
	if(to_sw == nullptr)
		throw Exception("output layer is not derived from LayerHW");
	
	const LayerHW_BP *from_sw = dynamic_cast<const LayerHW_BP *>(from);
	if(from_sw == nullptr)
		throw Exception("input layer is not derived from LayerHW_BP");
	
	getKernel("backpropBiasGrad")->evaluate(
	      cl::work_range(getOutputSize()), getOutputSize(),
	      from_sw->getInputError().getBuffer(), getBiasGrad().getBuffer()
	      );
	
	getKernel("backpropWeightGrad")->evaluate(
	      cl::work_range(getInputSize(), getOutputSize()), ivec2(getInputSize(), getOutputSize()),
	      from_sw->getInputError().getBuffer(), to_sw->getOutput().getBuffer(), getWeightGrad().getBuffer()
	      );
}

void ConnHW_BP::_backprop(Layer_BP *to, const Layer_BP *from)
{
	LayerHW_BP *to_sw = dynamic_cast<LayerHW_BP *>(to);
	if(to_sw == nullptr)
		throw Exception("output layer is not derived from LayerHW_BP");
	
	const LayerHW_BP *from_sw = dynamic_cast<const LayerHW_BP *>(from);
	if(from_sw == nullptr)
		throw Exception("input layer is not derived from LayerHW_BP");
	
	getKernel("backpropError")->evaluate(
	      cl::work_range(getInputSize()), getInputSize(), getOutputSize(),
	      from_sw->getInputError().getBuffer(), to_sw->getOutputError().getBuffer(),
	      getWeight().getBuffer()
	      );
	
	_backprop(static_cast<const Layer *>(to), from);
}

void ConnHW_BP::_bindQueue(cl::queue *queue)
{
	getWeightGrad().bindQueue(queue);
	getBiasGrad().bindQueue(queue);
	ConnHW::_bindQueue(queue);
}
