#pragma once

#include <exception>
#include <string>

namespace nn
{
class Exception : public std::exception
{
private:
	std::string _message;
	
public:
	Exception(const std::string &message)
	  : _message(message)
	{
		
	}
	
	virtual ~Exception() = default;
	
	const char *what() const noexcept
	{
		return _message.data();
	}
	
	const std::string &getMessage() const
	{
		return _message;
	}
};
}
