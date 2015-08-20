#pragma once

#include "buffer.hpp"

// TODO: rename 'output' to 'activation'

class Layer
{
public:
	typedef unsigned ID;
	
	class Buffer : public virtual ::Buffer
	{
	private:
		bool _validity = false;
		bool _zero = false;
		
	protected:
		Buffer() : Buffer(getSize()) {}
	public:
		Buffer(int size) : ::Buffer(size) {}
		virtual ~Buffer() = default;
		
		void setZero(bool z);
		bool isZero() const;
		
		void validate(bool v);
		bool isValid() const;
	};
	
private:
	ID _id;
	int _size;
	
protected:
	virtual void _update() = 0;
	
public:
	Layer(ID id, int size);
	virtual ~Layer() = default;
	
	ID getID();
	int getSize() const;
	
	virtual Buffer &getInput() = 0;
	virtual Buffer &getOutput() = 0;
	virtual const Buffer &getInput() const = 0;
	virtual const Buffer &getOutput() const = 0;
	
	void update();
};
