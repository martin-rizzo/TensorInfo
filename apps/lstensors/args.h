/*
| File    : args.h
| Purpose : The arguments of the `lstensors` command line
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 7, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef ARGS_H_
#define ARGS_H_
#include <iostream>
#include <string>

enum class Command {
    LIST_TENSORS,
    PRINT_TENSOR,
    PRINT_METADATA,
    EXTRACT_THUMBNAIL
};

enum class Format {
    HUMAN,
    PLAIN,
    JSON
};

struct Args
{
public:
    static Args from_main(int argc, char* argv[]);

public:
    const Command     command;
    const Format      format;
    const std::string input_file;
    const std::string prefix;
    const std::string tensor_name;
    const int         depth;
    const bool        no_color;
    const bool        help;
};
std::ostream& operator<<(std::ostream& os, const Args& args);

#endif // ARGS_H_
