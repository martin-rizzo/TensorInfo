/*
| File    : readerror.h
| Purpose : Enumeration of possible errors during read operations.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 11, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_READERROR_H
#define TIN_READERROR_H
namespace tin {

enum class ReadError {
    None,                   ///< No error occurred.
    FileNotFound,           ///< The specified file does not exist.
    InvalidFormat,          ///< The file format is invalid.
    UnsupportedVersion,     ///< The file format version is not supported by the library.
    HeaderTooLarge,         ///< The header size exceeds the maximum allowed value.
    MemoryAllocationFailed, ///< Failed to allocate memory for reading data.
    MissingData,            ///< A required field is missing or the file is unreadable at some point.
};


}      // namespace tin
#endif // TIN_READERROR_H
