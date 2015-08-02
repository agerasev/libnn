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
	Layer(ID id, int size);
	virtual ~Layer() = default;
	
	ID getID();
	int getSize() const;
	
	bool isValid() const;
	void setValidity(bool v);
	
	void write(const float *data);
	void read(float *data) const;
	void clear();
	void update();
};
}
