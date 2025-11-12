/*
| File    : gguf/gguf.h
| Purpose : Helper functions and constants for reading/writing GGUF files
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 11, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_GGUF_HELPER_H
#define TIN_GGUF_HELPER_H
#include <tin/common.h>
namespace tin {

//=============================== CONSTANTS ===============================//
// [https://github.com/ggml-org/ggml/blob/master/docs/gguf.md]

enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

//============================ GGUF FUNCTIONS =============================//

class GGUF
{
public:

    enum class METADATA_VALUE_TYPE {
        // The value is a 8-bit unsigned integer.
        UINT8 = 0,
        // The value is a 8-bit signed integer.
        INT8 = 1,
        // The value is a 16-bit unsigned little-endian integer.
        UINT16 = 2,
        // The value is a 16-bit signed little-endian integer.
        INT16 = 3,
        // The value is a 32-bit unsigned little-endian integer.
        UINT32 = 4,
        // The value is a 32-bit signed little-endian integer.
        INT32 = 5,
        // The value is a 32-bit IEEE754 floating point number.
        FLOAT32 = 6,
        // The value is a boolean.
        // 1-byte value where 0 is false and 1 is true.
        // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
        BOOL = 7,
        // The value is a UTF-8 non-null-terminated string, with length prepended.
        STRING = 8,
        // The value is an array of other values, with the length and type prepended.
        ///
        // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
        ARRAY = 9,
        // The value is a 64-bit unsigned little-endian integer.
        UINT64 = 10,
        // The value is a 64-bit signed little-endian integer.
        INT64 = 11,
        // The value is a 64-bit IEEE754 floating point number.
        FLOAT64 = 12,
    };


// READING INTEGERS
public:
    static Optional<int>           read_int8(std::istream& istream) noexcept;
    static Optional<unsigned>      read_uint8(std::istream& istream) noexcept;
    static Optional<int>           read_le_int16(std::istream& istream) noexcept;
    static Optional<unsigned>      read_le_uint16(std::istream& istream) noexcept;
    static Optional<int>           read_le_int32(std::istream& istream) noexcept;
    static Optional<unsigned>      read_le_uint32(std::istream& istream) noexcept;
    static Optional<long>          read_le_int64(std::istream& istream) noexcept;
    static Optional<unsigned long> read_le_uint64(std::istream& istream) noexcept;

// READING FLOAT-POINT NUMBERS
public:
    static Optional<float>  read_float32(std::istream& istream) noexcept;
    static Optional<double> read_float64(std::istream& istream) noexcept;

// READING OTHER VALUES
public:
    static Optional<String>         read_string(std::istream& istream) noexcept;
    static Optional<Vector<String>> read_string_array(std::istream& istream) noexcept;
};


}      // namespace tin
#endif // TIN_GGUF_HELPER_H
