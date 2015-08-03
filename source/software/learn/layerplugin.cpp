#include <nn/software/layer.hpp>
#include <nn/software/learn/layerplugin.hpp>

using namespace nn::sw;

LayerX<bp::LayerPlugin>::_init()
{
	_input_error = new float[getSize()];
	_output_error = new float[getSize()];
}

LayerX<bp::LayerPlugin>::_cleanup()
{
	delete[] _input_buffer;
	delete[] _output_buffer;
}
