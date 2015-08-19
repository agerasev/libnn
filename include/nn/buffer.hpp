#pragma once

class Buffer
{
private:
	int _size;
	bool _validity = false;
	bool _zero = false;
	
public:
	Buffer(int size);
	virtual ~Buffer() = default;
	
	int getSize() const;
	
	virtual void read(float *data) const = 0;
	virtual void write(const float *data) = 0;
	virtual void clear() = 0;
	
	virtual void copy(const Buffer &buffer) = 0;
	
	void setZero(bool z);
	bool isZero() const;
	
	void validate(bool v);
	bool isValid() const;
};
