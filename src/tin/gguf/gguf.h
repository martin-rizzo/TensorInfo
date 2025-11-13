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


class GGUF
{
public:

    // [https://github.com/ggml-org/ggml/blob/master/docs/gguf.md]
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
    static Optional<String> read_string(std::istream& istream) noexcept;
    static Optional<METADATA_VALUE_TYPE> read_value_type(std::istream& istream) noexcept;

// IGNORING VALUES
public:
    static void ignore_string(std::istream& istream) noexcept;
    static void ignore_array(std::istream& istream) noexcept;
};

}      // namespace tin


inline std::string_view
to_string( ::tin::GGUF::METADATA_VALUE_TYPE type ) {
    switch( type ) {
        case ::tin::GGUF::METADATA_VALUE_TYPE::UINT8:   return "uint8";
        case ::tin::GGUF::METADATA_VALUE_TYPE::INT8:    return "int8";
        case ::tin::GGUF::METADATA_VALUE_TYPE::UINT16:  return "uint16";
        case ::tin::GGUF::METADATA_VALUE_TYPE::INT16:   return "int16";
        case ::tin::GGUF::METADATA_VALUE_TYPE::UINT32:  return "uint32";
        case ::tin::GGUF::METADATA_VALUE_TYPE::INT32:   return "int32";
        case ::tin::GGUF::METADATA_VALUE_TYPE::FLOAT32: return "float32";
        case ::tin::GGUF::METADATA_VALUE_TYPE::BOOL:    return "bool";
        case ::tin::GGUF::METADATA_VALUE_TYPE::STRING:  return "string";
        case ::tin::GGUF::METADATA_VALUE_TYPE::ARRAY:   return "array";
        case ::tin::GGUF::METADATA_VALUE_TYPE::UINT64:  return "uint64";
        case ::tin::GGUF::METADATA_VALUE_TYPE::INT64:   return "int64";
        case ::tin::GGUF::METADATA_VALUE_TYPE::FLOAT64: return "float64";
        default:return "<unknown>";
    }
}

#endif // TIN_GGUF_HELPER_H
