#pragma once

#include<cmath>
#include<list>
#include<functional>

#include"wire.hpp"

template <typename T>
class Node
{
private:
	
	template <typename WT>
	class BaseSet
	{
		virtual void add(WT *wire) = 0;
		virtual void remove(WT *wire) = 0;
		
		/* iteration stops when function return true */
		virtual void forEach(std::function<bool(WT*)>) = 0;
		virtual void forEachConst(std::function<bool(const WT*)>) const = 0;
		
		virtual int getNumber() const = 0;
	};
	
public:
	
	class InputSet : public BaseSet<WireOutput<T>>
	{
		
	};
	
	class OutputSet : public BaseSet<WireInput<T>>
	{
		
	};
	
	virtual InputSet &inputs() = 0;
	virtual const InputSet &inputs() const = 0;
	
	virtual OutputSet &outputs() = 0;
	virtual const OutputSet &outputs() const = 0;
	
	virtual ~Node()
	{
		
	}
	
	virtual void perform() = 0;
};

template <typename T>
class TransferNode : public Node<T>
{
public:
	
	class Input
	{
	public:
		WireOutput<T> *wire;
		T factor;
		
		Input(WireOutput<T> *w, T f) :
		  wire(w), factor(f)
		{
			
		}
	};
	
	class Output
	{
	public:
		WireInput<T> *wire;
		
		Output(WireInput<T> *w) :
		  wire(w)
		{
			
		}
	};
	
	class InputSet : public Node<T>::InputSet
	{
	private:
		
		std::list<Input> inputs;
		
	public:
		
		void add(WireOutput<T> *wire) override
		{
			inputs.push_back(Input(wire,0.0));
		}
		
		void add(WireOutput<T> *wire, T factor = 0.0)
		{
			inputs.push_back(Input(wire,factor));
		}
		
		void remove(WireOutput<T> *wire) override
		{
			inputs.remove_if( [wire](const Input &elem)->bool { return (elem.wire == wire); } );
		}
		
		T getFactor(const WireOutput<T> *wire) const
		{
			for(const Input &i : inputs)
			{
				if(i.wire == wire)
				{
					return i.factor;
				}
			}
		}
		
		void setFactor(WireOutput<T> *wire, T factor)
		{
			for(Input &i : inputs)
			{
				if(i.wire == wire)
				{
					i.factor = factor;
				}
			}
		}
		
		void forEach(std::function<bool(WireOutput<T>*)> func) override
		{
			for(Input &i : inputs)
			{
				if(func(i.wire))
				{
					break;
				}
			}
		}

		void forEachConst(std::function<bool(const WireOutput<T>*)> func) const override
		{
			for(const Input &i : inputs)
			{
				if(func(i.wire))
				{
					break;
				}
			}
		}
		
		void forEachFactor(std::function<bool(WireOutput<T>*,T&)> func)
		{
			for(Input &i : inputs)
			{
				if(func(i.wire,i.factor))
				{
					break;
				}
			}
		}

		void forEachFactorConst(std::function<bool(const WireOutput<T>*,T)> func) const
		{
			for(const Input &i : inputs)
			{
				if(func(i.wire,i.factor))
				{
					break;
				}
			}
		}
		
		int getNumber() const override
		{
			return inputs.size();
		}
		
	};
	
	class OutputSet : public Node<T>::OutputSet
	{
	private:
		
		std::list<Output> outputs;
		
	public:
		
		void add(WireInput<T> *wire) override
		{
			outputs.push_back(Output(wire));
		}
	
		void remove(WireInput<T> *wire) override
		{
			outputs.remove_if( [wire](const Output &elem)->bool { return (elem.wire == wire); } );
		}
		
		/* iteration stops when function return true */
		void forEach(std::function<bool(WireInput<T>*)> func) override
		{
			for(const Output &o : outputs)
			{
				if(func(o.wire))
				{
					break;
				}
			}
		}
		
		void forEachConst(std::function<bool(const WireInput<T>*)> func) const override
		{
			for(const Output &o : outputs)
			{
				if(func(o.wire))
				{
					break;
				}
			}
		}
		
		virtual int getNumber() const
		{
			return outputs.size();
		}
		
	};
	
private:
	
	InputSet input_set;
	OutputSet output_set;
	std::function<T(T)> transmit;
	
	static T identityFunc(T arg)
	{
		return arg;
	}
	
public:
	
	TransferNode(std::function<T(T)> transmissionFunc = identityFunc)
	{
		transmit = transmissionFunc;
	}
	
	virtual InputSet &inputs() override
	{
		return input_set;
	}

	virtual const InputSet &inputs() const override
	{
		return input_set;
	}
	
	virtual OutputSet &outputs() override
	{
		return output_set;
	}

	virtual const OutputSet &outputs() const override
	{
		return output_set;
	}
	
	void perform() override
	{
		T sum = 0.0;
		input_set.forEachFactorConst([&sum](const WireOutput<T> *w, T f) -> bool
		{
			sum += f*w->get();
			return false;
		});
		
		sum = transmit(sum);
		
		output_set.forEach([sum](WireInput<T> *w) -> bool
		{
			w->set(sum);
			return false;
		});
	}
};
