#pragma once

#include <nn/net.hpp>

class Net_BP : public Net
{
public:
	Net_BP() = default;
	virtual ~Net_BP() = default;
	
	virtual void stepBackward();
};
