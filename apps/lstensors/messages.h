/*
| File    : messages.h
| Purpose : 
  spanish: simple coleccion de tipos de mensajes para el usuario
  english: simple collection of user messages
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef MESSAGES_H_
#define MESSAGES_H_
#include <vector>
#include <string_view>


/**
 * A simple collection of user messages.
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


#endif // MESSAGES_H_