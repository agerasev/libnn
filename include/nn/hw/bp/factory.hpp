#pragma once

#include <nn/hw/factory.hpp>

#include <nn/hw/bp/layerext.hpp>
#include <nn/hw/bp/conn.hpp>

class FactoryHW_BP : public FactoryHW
{
public:
	FactoryHW_BP(const std::string &kernel_file);
	virtual ~FactoryHW_BP() = default;
	
	virtual LayerHW_BP *newLayer(Layer::ID id, int size, int extension = LayerFunc::UNIFORM) override;
	virtual ConnHW_BP *newConn(Conn::ID id, int input_size, int output_size) override;
};
