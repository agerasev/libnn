#include <nn/hw/buffer.hpp>

BufferHW::BufferHW()
    : BufferHW(getSize(), this)
{
	
}

BufferHW::BufferHW(int size, const KitHW *kit)
    : Buffer(size), KitHW(kit), _buffer(size, kit->getContext())
{
	
}

BufferHW::~BufferHW()
{
	
}

void BufferHW::read(float *data) const
{
	_buffer.load_data(data);
}

void BufferHW::write(const float *data)
{
	_buffer.store_data(data);
}

void BufferHW::clear()
{
	cl::work_range range({getSize()});
	getKernel("fill")->evaluate(range, getSize(), getCLBuffer(), 0.0f);
}

cl::buffer_object *BufferHW::getBuffer()
{
	return &_buffer;
}

const cl::buffer_object *BufferHW::getBuffer() const
{
	return &_buffer;
}

void BufferHW::_bindQueue(cl::queue *queue)
{
	_buffer.bind_queue(queue);
}
