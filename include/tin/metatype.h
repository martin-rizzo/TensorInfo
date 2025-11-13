/*
| File    : metatype.h
| Purpose : Enumerate the different metadata types stored within model checkpoints.
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

/**
 * Enumerates the different metadata types stored within model checkpoints.
 * 
 * The TensorInfo library supports five main data types (boolean, integer,
 * unsigned, float, and string), but formats like "GGUF" allow for a broader
 * range of metadata types. To keep track of the original format, this enum
 * class is used, which can be accessed via:
 *  `tensorMap.metadata().get("name.of.metadata").original_metatype()`.
 */
enum class MetaType
{
        // Boolean
        BOOL,             ///< Original value is a 8-bit integer, zero or one. (use `as_bool()` to get the value).

        // Signed Integers
        INT8,             ///< Original value is a 8-bit signed integer. (use `as_integer()` to get the value).
        INT16,            ///< Original value is a 16-bit signed little-endian integer. (use `as_integer()` to get the value).
        INT32,            ///< Original value is a 32-bit signed little-endian integer. (use `as_integer()` to get the value).
        INT64,            ///< Original value is a 64-bit signed little-endian integer. (use `as_integer()` to get the value).

        // Unsigned Integers
        UINT8,            ///< Original value is a 8-bit unsigned integer. (use `as_unsigned()` to get the value).
        UINT16,           ///< Original value is a 16-bit unsigned little-endian integer. (use `as_unsigned()` to get the value).
        UINT32,           ///< Original value is a 32-bit unsigned little-endian integer. (use `as_unsigned()` to get the value).
        UINT64,           ///< Original value is a 64-bit unsigned little-endian integer. (use `as_unsigned()` to get the value).

        // Floating Points
        FLOAT32,          ///< Original value is a 32-bit IEEE754 floating point number. (use `as_float()` to get the value).
        FLOAT64,          ///< Original value is a 64-bit IEEE754 floating point number. (use `as_float()` to get the value).

        // Strings
        STRING,           ///< Original value is a string. (use `as_string()` to get the value).

        // Arrays is quite complex because GGUF even supports nested arrays!
        ARRAY_OF_BOOLS,   ///< Original value is an array of booleans.
        ARRAY_OF_INT8,    ///< Original value is an array of 8-bit signed integers.
        ARRAY_OF_INT16,   ///< Original value is an array of 16-bit signed little-endian integers.
        ARRAY_OF_INT32,   ///< Original value is an array of 32-bit signed little-endian integers.
        ARRAY_OF_INT64,   ///< Original value is an array of 64-bit signed little-endian integers.
        ARRAY_OF_UINT8,   ///< Original value is an array of 8-bit unsigned integers.
        ARRAY_OF_UINT16,  ///< Original value is an array of 16-bit unsigned little-endian integers.
        ARRAY_OF_UINT32,  ///< Original value is an array of 32-bit unsigned little-endian integers.
        ARRAY_OF_UINT64,  ///< Original value is an array of 64-bit unsigned little-endian integers.
        ARRAY_OF_FLOAT32, ///< Original value is an array of 32-bit IEEE754 floating point numbers.
        ARRAY_OF_FLOAT64, ///< Original value is an array of 64-bit IEEE754 floating point numbers.
        ARRAY_OF_STRINGS, ///< Original value is an array of strings.
        ARRAY_OF_ARRAYS,  ///< Original value is a nested array (!!).
};


}      // namespace tin
#endif // TIN_METATYPE_H_
