#pragma once

#include<cmath>
#include<list>

#include"wire.hpp"

template <typename T>
class Node
{	
public:
	
	virtual ~Node()
	{
		
	}
	
	virtual void addInput(WireOutput<T> *wire, T factor) = 0;
	virtual void addOutput(WireInput<T> *wire) = 0;

	virtual void removeInput(WireOutput<T> *wire) = 0;
	virtual void removeOutput(WireInput<T> *wire) = 0;
	
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

private:
	
	std::list<Input> inputs;
	std::list<Output> outputs;
	
public:
	
	void addInput(WireOutput<T> *wire, T factor = 0)
	{
		inputs.push_back(Input(wire,factor));
	}

	void addOutput(WireInput<T> *wire)
	{
		outputs.push_back(Output(wire));
	}

	void removeInput(WireOutput<T> *wire)
	{
		inputs.remove_if( [wire](const Input &elem)->bool { return (elem.wire == wire); } );
	}

	void removeOutput(WireInput<T> *wire)
	{
		outputs.remove_if( [wire](const Output &elem)->bool { return (elem.wire == wire); } );
	}
	
	void perform() override
	{
		T sum = 0.0;
		for(Input &i : inputs)
		{
			sum += i.factor*i.wire->get();
		}
		
		/* This function must be defined by user */
		sum = neural_tools::correct(sum);
		
		for(Output &o : outputs)
		{
			o.wire->set(sum);
		}
	}
	
	void vary()
	{
		/* TODO: Vary random factor in inputs */
	}
	
};
