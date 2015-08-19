#pragma once

#include <nn/bp/layerext.hpp>
#include <nn/sw/layerext.hpp>

template <int X>
class LayerExtSW_BP : public virtual LayerExtSW<X>, public virtual LayerExt_BP<X>
{
public:
	LayerExtSW_BP() = delete;
	virtual ~LayerExtSW_BP() = default;
};

template <>
class LayerExtSW_BP<EXT_NONE> : public virtual LayerExtSW<EXT_NONE>, public virtual LayerExt_BP<EXT_NONE>
{
public:
	LayerExtSW_BP(ID id, int size);
	virtual ~LayerExtSW_BP() = default;
};

template <>
class LayerExtSW_BP<EXT_SIGMOID> : public virtual LayerExtSW<EXT_SIGMOID>, public virtual LayerExt_BP<EXT_SIGMOID>
{
public:
	LayerExtSW_BP(ID id, int size);
	virtual ~LayerExtSW_BP() = default;
	
protected:
	virtual void _updateError() override;
};
