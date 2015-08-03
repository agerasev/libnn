#include <nn/sw/layer.hpp>

class SigmoidLayerSW : public virtual LayerSW
{
public:
	SigmoidLayerSW(ID id, int size);
	virtual ~SigmoidLayerSW() = default;

protected:
	static float _sigma(float a);
	virtual void _update() override;
};
