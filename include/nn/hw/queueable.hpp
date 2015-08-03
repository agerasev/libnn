#pragma once

#include <cl/queue.hpp>

class HWQueueable
{
private:
	cl_command_queue _queue;

protected:
	virtual void _bindQueue(cl_command_queue queue) = 0;
	
public:
	HWQueueable() = default;
	virtual ~HWQueueable() = default;
	
	void bindQueue(cl_command_queue queue)
	{
		_queue = queue;
		_bindQueue(queue);
	}
	
	void bindQueue(cl::queue &queue)
	{
		bindQueue(queue.get_cl_command_queue());
	}
	
	cl_command_queue getQueue() const
	{
		return _queue;
	}
};
