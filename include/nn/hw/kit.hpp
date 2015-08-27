#pragma once

#include <cl/context.hpp>
#include <cl/kernel.hpp>
#include <cl/map.hpp>
#include <cl/queue.hpp>

#include <string>

namespace cl
{
typedef cl::map<cl::kernel *> kernel_map;
}

class KitHW
{
private:
	cl::context *_context;
	cl::queue *_queue = nullptr;
	const cl::kernel_map *_kernels;
	
public:
	KitHW(cl::context *context, const cl::kernel_map *kernels);
	KitHW(const KitHW &kit);
	KitHW(const KitHW *kit);
	
	void bindQueue(cl::queue *queue);
	
	cl::context *getContext() const;
	cl::queue *getQueue() const;
	const cl::kernel_map *getKernelMap() const;
	cl::kernel *getKernel(const std::string &kernel_name) const;
	
protected:
	virtual void _bindQueue(cl::queue *queue);
};
