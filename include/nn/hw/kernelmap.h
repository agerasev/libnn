#pragma once

#include <string>

#include <cl/kernel.hpp>
#include <cl/map.hpp>

class KernelMapHW
{
private:
	const cl::map<cl::kernel *> &_kernels;
	
public:
	KernelMapHW(const cl::map<cl::kernel *> *kernels)
	    : _kernels(*kernels)
	{
		
	}
	
	cl::kernel *getKernel(const std::string &kernel_name)
	{
		return _kernels[kernel_name];
	}

	const cl::map<cl::kernel *> &getKernelMap()
	{
		return _kernels;
	}
};
