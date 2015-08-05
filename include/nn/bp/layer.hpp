#pragma once

#include <nn/layer.hpp>

class Layer_BP : public virtual Layer
{
public:
	Layer_BP() = default;
	virtual ~Layer_BP() = default;
	
	void writeError(const float *data);
	void readError(float *data) const;
	void updateError();
	void clearError();
	
protected:
	virtual void _writeError(const float *data) = 0;
	virtual void _readError(float *data) const = 0;
	virtual void _updateError() = 0;
	virtual void _clearError() = 0;
};
