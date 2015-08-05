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
			ll->clearError();
		}
	}
	for(std::pair<Conn::ID, Conn *> c : _conns)
	{
		Conn_BP *lc = dynamic_cast<Conn_BP *>(c.second);
		if(c != nullptr)
		{
			std::pair<Layer *, Layer *> p = _struct[lc->getID()];
			lc->backprop(p.first, p.second);
		}
	}
}
