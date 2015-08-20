#pragma once

class Buffer
{
private:
	int _size;
	
public:
	Buffer(int size);
	virtual ~Buffer() = default;
	
	int getSize() const;
	
	virtual void read(float *data) const = 0;
	virtual void write(const float *data) = 0;
	virtual void clear() = 0;
};
