/*
| File    : message.h
| Purpose : Simple class to display messages to the user in the console.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef MESSAGE_H_
#define MESSAGE_H_
#include <string_view> // for std::string_view
#include <vector>      // for std::vector


/**
 * Simple class to display messages to the user in the console.
 *
 * This class provides static methods to display different types of messages:
 *  - Warnings    : Informative messages that do not interrupt the flow of the program.
 *  - Errors      : Issues encountered during execution that may require attention but are not critical.
 *  - Fatal Errors: Critical issues that warrant immediate termination of the program.
 */
class Message
{
public:
    static void warning(std::string_view message);
    static void error(std::string_view message);
    static void fatal_error(std::string_view message, const std::vector<std::string_view>& informativeMessages = {}, int exitCode = 1);
};


#endif // MESSAGE_H_
