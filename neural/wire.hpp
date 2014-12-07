#pragma once

template <typename T>
class WireInput
{
public:
	virtual void set(T st) = 0;
	virtual ~WireInput()
	{
		
	}
};

template <typename T>
class WireOutput
{
public:
	virtual T get() const = 0;
	virtual ~WireOutput()
	{
		
	}
};

template <typename T>
class Wire : public WireInput<T>, public WireOutput<T>
{
public:
	virtual ~Wire()
	{
		
	}
};

template <typename T>
class CapableWire : public Wire<T>
{
private:
	
	T in;
	T out;
	
public:
	
	CapableWire() :
	  in(0), out(0)
	{
		
	}

	void set(T st) override
	{
		in = st;
	}
	
	T get() const override
	{
		return out;
	}
	
	void update()
	{
		out = in;
	}
	
};
