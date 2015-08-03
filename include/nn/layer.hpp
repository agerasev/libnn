#pragma once

class Layer
{
public:
	typedef unsigned ID;
	
private:
	ID _id;
	int _size;
	
	bool _zero_in = true;
	bool _zero_out = true;
	
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
	
	bool isZero() const;
	void setZero(bool v);
	
	void write(const float *data);
	void read(float *data) const;
	void clear();
	void update();
};
