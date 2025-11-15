/*
| File    : storagetype.h
| Purpose : Enumerate all possible storage types for checkpoint metadata.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 12, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_STORAGETYPE_H_
#define TIN_STORAGETYPE_H_
namespace tin {


/**
 * Enumerates all possible storage types for checkpoint metadata.
 * 
 * While the TensorInfo library supports five primary data types (boolean,
 * integer, unsigned integer, float, and string), formats such as "GGUF"
 * allow a wider range of checkpoint metadata types. This enum facilitates 
 * identification of the storage type utilized when storing metadata in
 * files. The specific storage type can be accessed via:
 *  `tensorMap.metadata().get("name.of.metadata").storage_type()`.
 * 
 * @see MetaValue::storage_type()
 * @see MetaValue::as_boolean()
 * @see MetaValue::as_integer()
 * @see MetaValue::as_unsigned()
 * @see MetaValue::as_double()
 * @see MetaValue::as_string()
 */
enum class StorageType
{
        // Boolean
        BOOL,             ///< Stored value is a 8-bit integer, zero or one. (use `as_bool()` to get the value).

        // Signed Integers
        INT8,             ///< Stored value is a 8-bit signed integer. (use `as_integer()` to get the value).
        INT16,            ///< Stored value is a 16-bit signed little-endian integer. (use `as_integer()` to get the value).
        INT32,            ///< Stored value is a 32-bit signed little-endian integer. (use `as_integer()` to get the value).
        INT64,            ///< Stored value is a 64-bit signed little-endian integer. (use `as_integer()` to get the value).

        // Unsigned Integers
        UINT8,            ///< Stored value is a 8-bit unsigned integer. (use `as_unsigned()` to get the value).
        UINT16,           ///< Stored value is a 16-bit unsigned little-endian integer. (use `as_unsigned()` to get the value).
        UINT32,           ///< Stored value is a 32-bit unsigned little-endian integer. (use `as_unsigned()` to get the value).
        UINT64,           ///< Stored value is a 64-bit unsigned little-endian integer. (use `as_unsigned()` to get the value).

        // Floating Points
        FLOAT32,          ///< Stored value is a 32-bit IEEE754 floating point number. (use `as_float()` to get the value).
        FLOAT64,          ///< Stored value is a 64-bit IEEE754 floating point number. (use `as_float()` to get the value).

        // Strings
        STRING,           ///< Stored value is a string. (use `as_string()` to get the value).

        // Arrays is quite complex because GGUF even supports nested arrays!
        ARRAY_OF_BOOLS,   ///< Stored value is an array of booleans.
        ARRAY_OF_INT8,    ///< Stored value is an array of 8-bit signed integers.
        ARRAY_OF_INT16,   ///< Stored value is an array of 16-bit signed little-endian integers.
        ARRAY_OF_INT32,   ///< Stored value is an array of 32-bit signed little-endian integers.
        ARRAY_OF_INT64,   ///< Stored value is an array of 64-bit signed little-endian integers.
        ARRAY_OF_UINT8,   ///< Stored value is an array of 8-bit unsigned integers.
        ARRAY_OF_UINT16,  ///< Stored value is an array of 16-bit unsigned little-endian integers.
        ARRAY_OF_UINT32,  ///< Stored value is an array of 32-bit unsigned little-endian integers.
        ARRAY_OF_UINT64,  ///< Stored value is an array of 64-bit unsigned little-endian integers.
        ARRAY_OF_FLOAT32, ///< Stored value is an array of 32-bit IEEE754 floating point numbers.
        ARRAY_OF_FLOAT64, ///< Stored value is an array of 64-bit IEEE754 floating point numbers.
        ARRAY_OF_STRINGS, ///< Stored value is an array of strings.
        ARRAY_OF_ARRAYS,  ///< Stored value is a nested array (!!).
};


}      // namespace tin
#endif // TIN_STORAGETYPE_H_
