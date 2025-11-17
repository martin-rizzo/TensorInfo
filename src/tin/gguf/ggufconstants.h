/*
| File    : gguf/ggufconstants.h
| Purpose : Constants used internally in GGUF files.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 16, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_GGUFCONSTANTS_H
#define TIN_GGUFCONSTANTS_H
namespace tin {

/**
 * Enumerates the types of data that can be found in GGML tensors.
 *
 * Reference:
 * https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#file-structure
 */
enum class GGMLType {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // Q4_2 = 4, support has been removed
    // Q4_3 = 5, support has been removed
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    // Q4_0_4_4 = 31, support has been removed from gguf files
    // Q4_0_4_8 = 32,
    // Q4_0_8_8 = 33,
    TQ1_0   = 34,
    TQ2_0   = 35,
    // IQ4_NL_4_4 = 36,
    // IQ4_NL_4_8 = 37,
    // IQ4_NL_8_8 = 38,
    MXFP4   = 39, // MXFP4 (1 block)
    COUNT   = 40,
};

/**
 * Enumerates the types of values that can be stored as metadata within GGUF files.
 *
 * It covers a range of integer and floating-point data types, strings, boolean
 * values, arrays, and nested arrays. Each type specifies how the data is encoded
 * in the file.
 * 
 * Reference:
 * https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#file-structure
 */
enum class GGUFMetadataValueType {
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


}      // namespace tin
#endif // TIN_GGUFCONSTANTS_H