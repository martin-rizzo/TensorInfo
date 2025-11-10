/*
| File    : lstensors_command.h
| Purpose : The `lstensors` command line tool.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 7, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef LSTENSORS_COMMAND_H_
#define LSTENSORS_COMMAND_H_
#include "lstensors_args.h"


class LsTensorsCommand
{
// MAIN
public:
    LsTensorsCommand(const LsTensorsArgs& args);
    [[nodiscard]] int run();


// HELPERS
public:
    void print_help() const;


// IMPLEMENTATION
private:
    const LsTensorsArgs _args;
};

#endif // LSTENSORS_COMMAND_H_