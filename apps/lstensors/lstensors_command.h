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
#include <tin/readerror.h>  // for tin::ReadError
#include <tin/tensormap.h>  // for tin::TensorMap
#include "lstensors_args.h" // for LsTensorsArgs
using tin::TensorMap;
using tin::ReadError;

class LsTensorsCommand
{
// MAIN
public:
    LsTensorsCommand(const LsTensorsArgs& args);
    [[nodiscard]] int run();


// SUBCOMMANDS
public:
    void list_tensors_columns(const TensorMap& tensorMap) const;
    void list_tensors_csv(const TensorMap& tensorMap, bool includeHeaders=true) const;
    void list_metadata(const TensorMap& tensorMap) const;
    void print_metadata(const TensorMap& tensorMap, const std::string& key) const;

// HELPERS
public:
    static void print_help();
    [[noreturn]] static void fatal_read_error( ReadError error );


// IMPLEMENTATION
private:
    const LsTensorsArgs _args;
};

#endif // LSTENSORS_COMMAND_H_