#include <nn/exception.hpp>

Exception::Exception(const std::string &message)
	: _message(message)
{
	
}

const char *Exception::what() const noexcept
{
	return _message.data();
}

const std::string &Exception::getMessage() const
{
	return _message;
}

