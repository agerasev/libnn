#include <nn/sw/bp/layerext.hpp>

LayerExtSW_BP<EXT_NONE>::LayerExtSW_BP(ID id, int size)
    : Layer(id, size), LayerSW(id, size)
{
	
}

virtual void _updateError() override;
