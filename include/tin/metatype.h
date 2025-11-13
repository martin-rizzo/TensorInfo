/*
| File    : metatype.h
| Purpose : Enumeration of types of values stored in the metadata of a model checkpoint.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 12, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_METATYPE_H_
#define TIN_METATYPE_H_
namespace tin {


enum class MetaType {

        // The value is a boolean.
        // 1-byte value where 0 is false and 1 is true.
        // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
        BOOL,

        INT8, //< Value is a 8bit signed integer. `metadata().insert_integer(k, value, MetaType::INT8)`

        INT16, //< Value is 16bit signed LE integer. `metadata().insert_integer(k, value, MetaType::INT16)`

        // The value is a 32-bit signed little-endian integer.
        INT32,

        // The value is a 64-bit signed little-endian integer.
        INT64,

        // The value is a 8-bit unsigned integer.
        UINT8,
        
        // The value is a 16-bit unsigned little-endian integer.
        UINT16,

        // The value is a 32-bit unsigned little-endian integer.
        UINT32 = 4,

        // The value is a 64-bit unsigned little-endian integer.
        UINT64 = 10,

        // The value is a 32-bit IEEE754 floating point number.
        FLOAT32 = 6,

        // The value is a 64-bit IEEE754 floating point number.
        FLOAT64 = 12,

        // The value is a UTF-8 non-null-terminated string, with length prepended.
        STRING = 8,
        // The value is an array of other values, with the length and type prepended.
        ///
        // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
        STRING_ARRAY = 9,



};


}      // namespace tin
#endif // TIN_METATYPE_H_
