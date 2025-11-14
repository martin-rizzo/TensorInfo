/*
| File    : lstensors_args.h
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
#ifndef LSTENSORS_ARGS_H_
#define LSTENSORS_ARGS_H_
#include <iostream>
#include <string>

enum class Command {
    LIST_TENSORS,
    LIST_METADATA,
    EXTRACT_THUMBNAIL
};
inline std::string to_string(Command command) {
    switch (command) {
        case Command::LIST_TENSORS     : return "Command::LIST_TENSORS";
        case Command::LIST_METADATA    : return "Command::LIST_METADATA";
        case Command::EXTRACT_THUMBNAIL: return "Command::EXTRACT_THUMBNAIL";
        default: return "<unknown>";
    }
}

enum class Format {
    HUMAN,
    PLAIN,
    JSON
};
inline std::string to_string(Format format) {
    switch (format) {
        case Format::HUMAN: return "Format::HUMAN";
        case Format::PLAIN: return "Format::PLAIN";
        case Format::JSON : return "Format::JSON";
        default: return "<unknown>";
    }
}

enum class UseColor {
    AUTO,
    NEVER,
    ALWAYS
};
inline std::string to_string(UseColor color) {
    switch (color) {
        case UseColor::AUTO : return "UseColor::AUTO";
        case UseColor::NEVER: return "UseColor::NEVER";
        case UseColor::ALWAYS: return "UseColor::ALWAYS";
        default: return "<unknown>";
    }
}

inline std::string to_string(bool value) {
    return value ? "true" : "false";
}

struct LsTensorsArgs
{
// CONSTRUCTION/DESTRUCTION
public:
    LsTensorsArgs(int argc, char* argv[]);
    LsTensorsArgs() = default;
    LsTensorsArgs(const LsTensorsArgs&) = default;
    LsTensorsArgs(LsTensorsArgs&&) noexcept = default;
    ~LsTensorsArgs() = default;

// PUBLIC MEMBERS
public:
    Command     command     = Command::LIST_TENSORS;
    std::string filename    = "";             ///< The name of the file to read
    std::string name        = "";             ///< The name of the tensor to print
    std::string prefix      = "";             ///< Only print tensors with this prefix
    int         depth       = 0;              ///< The depth of the tree to print
    Format      format      = Format::HUMAN;  ///< Output format
    UseColor    use_color   = UseColor::AUTO; ///< Whether to use color in output
    bool        help        = false;          ///< true = print usage and exit
};
std::ostream& operator<<(std::ostream& os, const LsTensorsArgs& args);

#endif // LSTENSORS_ARGS_H_
