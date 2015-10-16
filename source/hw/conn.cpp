#include <nn/hw/conn.hpp>

#include <nn/exception.hpp>
#include <la/vec.hpp>

#include <cstdlib>

ConnHW::ConnHW(ID id, int input_size, int output_size, int weight_size, int bias_size, const KitHW *kit)
    : Conn(id, input_size, output_size),
      KitHW(kit),
      _weight(weight_size, kit), 
      _bias(bias_size, kit)
{
	int width = (getInputSize() - 1)/REDUCE_FACTOR + 1;
	while(width > 1) 
	{
		_reduce_buffers.push_back(new ConnHW::BufferHW(width*getOutputSize(), kit));
		width = (width - 1)/REDUCE_FACTOR + 1;
	}
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
	for(ConnHW::BufferHW *buffer : _reduce_buffers) {
		delete buffer;
	}
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
	
#ifndef NN_NO_OPTIM
	if(_reduce_buffers.size() == 0)
#endif
	{
		getKernel("transmit")->evaluate(
					cl::work_range(getOutputSize()), getInputSize(), getOutputSize(),
					input->getOutput().getBuffer(), output->getInput().getBuffer(), 
					_weight.getBuffer(), _bias.getBuffer()
					);
	}
#ifndef NN_NO_OPTIM
	else
	{
		int ix = getInputSize(), iy = getOutputSize();
		int ixr = ((getInputSize() - 1)/REDUCE_FACTOR + 1);
		
		cl::work_range init_range(ixr, iy);
		getKernel("transmit_reduce_init")->evaluate(
					init_range, ix, ivec2(ixr, iy), input->getOutput().getBuffer(), 
					_reduce_buffers[0]->getBuffer(), _weight.getBuffer()
					);
		
		for(int i = 0; i < (int) _reduce_buffers.size() - 1; ++i) 
		{
			int ixrn = ((ixr - 1)/REDUCE_FACTOR + 1);
			getKernel("transmit_reduce")->evaluate(
						cl::work_range(ixrn,iy), ixr, ivec2(ixrn,iy), 
						_reduce_buffers[i]->getBuffer(), _reduce_buffers[i + 1]->getBuffer()
						);
			ixr = ixrn;
		}
		
		getKernel("transmit_reduce_finalize")->evaluate(
					cl::work_range(iy), ixr, iy, 
					_reduce_buffers[_reduce_buffers.size() - 1]->getBuffer(), 
					output->getInput().getBuffer(), _bias.getBuffer()
					);
	}
#endif
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
