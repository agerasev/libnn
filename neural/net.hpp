#pragma once

#include<list>
#include<map>
#include<exception>

#include"node.hpp"
#include"wire.hpp"
#include"image.hpp"

template <typename T>
class Net
{
public:
	
	class BuildException : public std::exception
	{
	public:
		const char *what() const noexcept override
		{
			return "Net build exception";
		}
	};
	
	virtual ~Net()
	{
		
	}
	
	virtual std::list<WireInput<T>*> getInputs() = 0;
	virtual std::list<WireOutput<T>*> getOutputs() = 0;
	
	virtual void setImage(const NetImage<T> &image) throw(BuildException) = 0;
	virtual NetImage<T> getImage() const = 0;
};

template <typename T>
class SynchronizedNet : public Net<T>
{
private:
	
	std::list<TransferNode<T>*> nodes;
	std::list<CapableWire<T>*> wires;
	std::list<CapableWire<T>*> inputs;
	std::list<CapableWire<T>*> outputs;
	
public:
	
	SynchronizedNet()
	{
		
	}
	
	~SynchronizedNet()
	{
		clear();
	}
	
	void clear()
	{
		for(TransferNode<T> *n : nodes)
		{
			delete n;
		}
		nodes.clear();
		
		for(CapableWire<T> *w : wires)
		{
			delete w;
		}
		wires.clear();
		
		for(CapableWire<T> *i : inputs)
		{
			delete i;
		}
		inputs.clear();
		
		for(CapableWire<T> *o : outputs)
		{
			delete o;
		}
		outputs.clear();
	}
	
	virtual std::list<WireInput<T>*> getInputs() override
	{
		std::list<WireInput<T>*> list;
		for(CapableWire<T> *i : inputs)
		{
			list.push_back(i);
		}
		return list;
	}
	
	virtual std::list<WireOutput<T>*> getOutputs() override
	{
		std::list<WireOutput<T>*> list;
		for(CapableWire<T> *i : outputs)
		{
			list.push_back(i);
		}
		return list;
	}
	
	void perform()
	{
		for(TransferNode<T> *n : nodes)
		{
			n->perform();
		}
	}
	
	void update()
	{
		for(CapableWire<T> *w : wires)
		{
			w->update();
		}
		
		for(CapableWire<T> *i : inputs)
		{
			i->update();
		}
		
		for(CapableWire<T> *o : outputs)
		{
			o->update();
		}
	}
	
	void setImage(const NetImage<T> &image) throw(typename Net<T>::BuildException) override
	{
		clear();
		
		/* Creating nodes and caching */
		std::map<int,TransferNode<T>*> node_map;
		std::map<int,CapableWire<T>*> input_map;
		std::map<int,CapableWire<T>*> output_map;
		for(const NodeImage<T> &ni : image.nodes)
		{
			TransferNode<T> *node = new TransferNode<T>();
			node_map.insert(std::pair<int,TransferNode<T>*>(ni.id,node));
			nodes.push_back(node);
		}
		for(const InputImage &ii : image.inputs)
		{
			CapableWire<T> *input = new CapableWire<T>();
			input_map.insert(std::pair<int,CapableWire<T>*>(ii.id,input));
			inputs.push_back(input);
		}
		for(const OutputImage &oi : image.outputs)
		{
			CapableWire<T> *output = new CapableWire<T>();
			output_map.insert(std::pair<int,CapableWire<T>*>(oi.id,output));
			outputs.push_back(output);
		}
		
		/* Connecting nodes */
		for(const NodeImage<T> &ni : image.nodes)
		{
			TransferNode<T> *node = node_map.find(ni.id)->second;
			for(const typename NodeImage<T>::Input &nii : ni.inputs)
			{
				auto node_iter = node_map.find(nii.id);
				if(node_iter != node_map.end())
				{
					CapableWire<T> *wire = new CapableWire<T>();
					node->inputs().add(wire,nii.factor);
					node_iter->second->outputs().add(wire);
					wires.push_back(wire);
					continue;
				}
				auto input_iter = input_map.find(nii.id);
				if(input_iter != input_map.end())
				{
					node->inputs().add(input_iter->second,nii.factor);
					continue;
				}
				/* Need to say about error here */
			}
			for(const typename NodeImage<T>::Output &nio : ni.outputs)
			{
				auto output_iter = output_map.find(nio.id);
				if(output_iter != output_map.end())
				{
					node->outputs().add(output_iter->second);
					continue;
				}
			}
		}
	}
	
	NetImage<T> getImage() const override
	{
		NetImage<T> image;
		/* TODO: fill image */
		return image;
	}
	
};
