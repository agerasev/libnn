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
	Exception(const std::string &message);
	virtual ~Exception() = default;
	
	virtual const char *what() const noexcept override;
	const std::string &getMessage() const;
};
}
