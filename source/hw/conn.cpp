#include <nn/hw/conn.hpp>

#include <nn/exception.hpp>

#include <cstdlib>

ConnHW::ConnHW(ID id, int input_size, int output_size, int weight_size, int bias_size, const KitHW *kit)
    : Conn(id, input_size, output_size),
      KitHW(kit),
      _weight(weight_size, kit), 
      _bias(bias_size, kit)
{
	
}

ConnHW::ConnHW()
    : ConnHW(
        getID(), getInputSize(), getOutputSize(),
        static_cast<const KitHW *>(this)
        )
{
	
}

ConnHW::ConnHW(ID id, int input_size, int output_size, const KitHW *kit)
    : ConnHW(id, input_size, output_size, input_size*output_size, output_size, kit)
{
	
}

ConnHW::~ConnHW()
{
	
}
        

ConnHW::BufferHW &ConnHW::getWeight()
{
	return _weight;
}

ConnHW::BufferHW &ConnHW::getBias()
{
	return _bias;
}

const ConnHW::BufferHW &ConnHW::getWeight() const
{
	return _weight;
}

const ConnHW::BufferHW &ConnHW::getBias() const
{
	return _bias;
}

void ConnHW::_transmit(const Layer *from, Layer *to) const
{
	const LayerHW *input = dynamic_cast<const LayerHW *>(from);
	if(input == nullptr)
		throw Exception("input layer is not derived from LayerHW");
	
	LayerHW *output = dynamic_cast<LayerHW *>(to);
	if(output == nullptr)
		throw Exception("output layer is not derived from LayerHW");
	
	cl::work_range range(getOutputSize());
	getKernel("transmit")->evaluate(
				range, getInputSize(), getOutputSize(),
				input->getOutput().getBuffer(), output->getInput().getBuffer(), 
				_weight.getBuffer(), _bias.getBuffer()
				);
}

void ConnHW::_bindQueue(cl::queue *queue)
{
	_weight.bindQueue(queue);
	_bias.bindQueue(queue);
}

void ConnHW::BufferHW::randomize(float range)
{
	// TODO: randomize on GPU
	int size = getSize();
	float *data = new float[size];
	for(int i = 0; i < size; ++i)
	{
		data[i] = range*(float(rand())/RAND_MAX - 0.5f);
	}
	getBuffer()->store_data(data);
	delete[] data;
}
