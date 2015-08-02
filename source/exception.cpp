#include <nn/exception.hpp>

nn::Exception::Exception(const std::string &message)
	: _message(message)
{
	
}

const char *nn::Exception::what() const noexcept
{
	return _message.data();
}

const std::string &nn::Exception::getMessage() const
{
	return _message;
}

