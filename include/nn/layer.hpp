#pragma once

namespace nn
{
class Layer
{
public:
	typedef unsigned ID;
	
private:
	ID _id;
	int _size;
	
	bool _current_valid = false;
	bool _next_valid = false;
	
protected:
	virtual void _write(const float *data) = 0;
	virtual void _read(float *data) const = 0;
	virtual void _clear() = 0;
	virtual void _update() = 0;
	
public:
	Layer(ID id, int size)
	  : _id(id), _size(size)
	{
		
	}
	
	virtual ~Layer() = default;
	
	ID getID()
	{
		return _id;
	}
	
	int getSize() const
	{
		return _size;
	}
	
	bool isValid() const
	{
		return _current_valid;
	}
	
	void setValidity(bool v)
	{
		_next_valid = v;
	}
	
	void write(const float *data)
	{
		_write(data);
		setValidity(true);
	}

	void read(float *data) const
	{
		_read(data);
	}
	
	void clear()
	{
		_clear();
		setValidity(false);
	}
	
	void update()
	{
		_update();
		_current_valid = _next_valid;
		_next_valid = false;
	}
};
}
