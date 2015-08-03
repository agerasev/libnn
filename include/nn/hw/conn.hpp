#pragma once

#include <cl/map.hpp>
#include <cl/kernel.hpp>
#include <cl/buffer_object.hpp>

#include "queueable.hpp"

#include <nn/exception.hpp>
#include <nn/conn.hpp>

#include "layer.hpp"

class ConnHW : public virtual Conn, public HWQueueable
{
private:
	cl::map<cl::kernel *> &_kernels;
	cl::buffer_object *_weight_buffer;
	cl::buffer_object *_bias_buffer;
	
protected:
	ConnHW(
	    ID id, int input_size, int output_size, 
	    cl_context context, cl::map<cl::kernel *> &kernels, 
	    int weight_size, int bias_size
	    );
	
	virtual void _bindQueue(cl_command_queue queue) override;
	virtual void _transmit(const Layer *from, Layer *to) const override;
	
public:
	ConnHW(
	    ID id, int input_size, int output_size, 
	    cl_context context, cl::map<cl::kernel *> &kernels
	    );
	virtual ~ConnHW();
	
	cl::buffer_object *getWeight();
	const cl::buffer_object *getWeight() const;
	cl::buffer_object *getBias();
	const cl::buffer_object *getBias() const;
	
	virtual void readWeight(float *data) const override;
	virtual void readBias(float *data) const override;
	virtual void writeWeight(const float *data) override;
	virtual void writeBias(const float *data) override;
};
