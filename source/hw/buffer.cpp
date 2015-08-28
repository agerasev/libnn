#include <nn/hw/buffer.hpp>

BufferHW::BufferHW()
    : BufferHW(getSize(), static_cast<KitHW *>(this))
{
	
}

BufferHW::BufferHW(int size, const KitHW *kit)
    : Buffer(size), KitHW(kit), _buffer(*kit->getContext(), size)
{
	
}

BufferHW::~BufferHW()
{
	
}

void BufferHW::read(float *data) const
{
	_buffer.load_data(data);
	getQueue()->flush();
}

void BufferHW::write(const float *data)
{
	_buffer.store_data(data);
	getQueue()->flush();
}

void BufferHW::clear()
{
	cl::work_range range({unsigned(getSize())});
	getKernel("fill")->evaluate(range, getSize(), getBuffer(), 0.0f);
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
	_buffer.bind_queue(queue->get_cl_command_queue());
}
