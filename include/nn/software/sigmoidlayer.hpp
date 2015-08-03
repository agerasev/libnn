#include <nn/software/layer.hpp>

namespace nn
{
namespace sw
{
class SigmoidLayer : public nn::sw::Layer
{
public:
	SigmoidLayer(ID id, int size);
	virtual ~SigmoidLayer() = default;

protected:
	virtual void _update() override;
};
}
}
