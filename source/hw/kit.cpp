#include <nn/hw/kit.hpp>

KitHW::KitHW(cl::context *context, const cl::kernel_map *kernels)
    : _context(context), _kernels(kernels)
{
	
}

KitHW::KitHW(const KitHW &kit)
    : _context(kit._context), _queue(kit._queue), _kernels(kit._kernels)
{
	_bindQueue(_queue);
}

KitHW(const KitHW *kit)
    : KitHW(*kit)
{
	
}

void KitHW::bindQueue(cl::queue *queue)
{
	_queue = queue;
	_bindQueue(queue);
}

cl::context *KitHW::getContext() const
{
	return _context;
}

cl::queue *KitHW::getQueue() const
{
	return _queue;
}

const cl::kernel_map *KitHW::getKernelMap() const
{
	return _kernels;
}

cl::kernel *KitHW::getKernel(const std::string &kernel_name) const
{
	return _kernels[kernel_name];
}

void KitHW::_bindQueue(cl::queue *queue)
{
	
}
