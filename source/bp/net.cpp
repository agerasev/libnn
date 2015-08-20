#include <nn/bp/net.hpp>

#include <nn/bp/layer.hpp>
#include <nn/bp/conn.hpp>

void Net_BP::stepBackward()
{
	for(std::pair<Layer::ID, Layer *> l : _layers)
	{
		Layer_BP *ll = dynamic_cast<Layer_BP *>(l.second);
		if(ll != nullptr)
		{
			ll->updateError();
		}
	}
	for(std::pair<Conn::ID, Conn *> c : _conns)
	{
		Conn_BP *lc = dynamic_cast<Conn_BP *>(c.second);
		if(lc != nullptr)
		{
			std::pair<Layer::ID, Layer::ID> p = _struct[lc->getID()];
			lc->backprop(_layers[p.first], _layers[p.second]);
		}
	}
}

void Net_BP::commitGrad(float delta)
{
	for(std::pair<Conn::ID, Conn *> c : _conns)
	{
		Conn_BP *lc = dynamic_cast<Conn_BP *>(c.second);
		if(lc != nullptr)
		{
			lc->commitGrad(delta);
		}
	}
}
