#pragma once

#include "buffer.hpp"

class Layer
{
public:
	typedef unsigned ID;
	
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
