#pragma once

#include<list>

template <typename T>
class NodeImage
{
public:
	class Input
	{
	public:
		int id;
		T factor;
		Input(int c_id, T c_factor) :
		  id(c_id), factor(c_factor)
		{
			
		}
	};
	
	class Output
	{
	public:
		int id;
		Output(int c_id) : id(c_id)
		{
			
		}
	};
	
public:
	int id;
	std::list<Input> inputs;
	std::list<Output> outputs;
	
public:
	NodeImage(int c_id)
	{
		id = c_id;
	}
};

class InputImage
{
public:
	int id;
	InputImage(int c_id) : id(c_id)
	{
		
	}
};

class OutputImage
{
public:
	int id;
	OutputImage(int c_id) : id(c_id)
	{
		
	}
};

template <typename T>
class NetImage
{
public:
	std::list<NodeImage<T>> nodes;
	std::list<InputImage> inputs;
	std::list<OutputImage> outputs;
};
