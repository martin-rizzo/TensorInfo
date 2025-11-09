/*
| File    : dtype.h
| Purpose : Enumerate different data types used for tensor storage.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 7, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_DTYPE_H_
#define TIN_DTYPE_H_
#include <type_traits>  // for std::is_same_V
#include <string_view>  // for std::string_view
#include <ostream>      // for std::ostream
#include <format>       // for std::formatter
namespace tin {


/**
 * Enumerates different data types used for tensor storage.
 *
 * This enumeration defines various data types that represent different
 * precision and size formats for storing tensors. It includes floating-point,
 * integer, boolean, and specific quantized formats tailored for efficient
 * storage and computation.
 */
enum class DType
{
    F64,      // 64-bit floating point
    F32,      // 32-bit floating point
    F16,      // 16-bit floating point (half precision)
    BF16,     // Brain floating point 
    F8_E8M0,  // F8_E8M0 [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf]
    F8_E5M2,  // FP8     [https://arxiv.org/pdf/2209.05433.pdf]
    F8_E4M3,  // FP8     [https://arxiv.org/pdf/2209.05433.pdf]
    F6_E3M2,  // MXF6    [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf]
    F6_E2M3,  // MXF6    [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf]
    F4,       // MXF4    [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf]
    I64,      // 64-bit signed integer
    I32,      // 32-bit signed integer
    I16,      // 16-bit signed integer
    I8,       // 8-bit signed integer
    U64,      // 64-bit unsigned integer
    U32,      // 32-bit unsigned integer
    U16,      // 16-bit unsigned integer
    U8,       // 8-bit unsigned integer
    BOOL,     // Boolean value

    //== K-QUANTS ===========================================================
    //[https://github.com/ggml-org/llama.cpp/pull/1684]
    //[https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py]

    Q6_K,     // "type-0" 6-bit. 16 blocks of 16 weights. Scales quantized with 8 bits.
              // Q4[128b] + Q2[64b] + scales_Q8[16b] + sizeof(float16) = 210 bytes (6.5625 bpw)

    Q5_K,     // "type-1" 5-bit. 8 blocks of 32 weights. Scales:Mins quantized with 6 bits.
              // sizeof(float16) + sizeof(float16) + scales_mins_Q6[12b] + Q1[32b] + Q4[128b] = 176 bytes (5.5 bpw)

    Q4_K,     // "type-1" 4-bit. 8 blocks of 32 weights. Scales:Mins quantized with 6 bits.
              // sizeof(float16) + sizeof(float16) + scales_mins_Q6[12b] + Q4[128b] = 144 bytes (4.5 bpw)

    Q3_K,     // "type-0" 6-bits. 16 blocks of 16 weights. Scales quantized with 6 bits.
              //  Q1[32b] + Q2[64b] + scales_Q6[12b] + sizeof(float16) = 110 bytes (3.4375 bpw)

    Q2_K,     // "type-1" 2-bit. 16 blocks of 16 weight. Scales:Mins quantized with 4 bits.
              // scales_mins_Q4[16b] + Q2[64b], sizeof(float16) + sizeof(float16) = 84 bytes (2.625)

    //== I-QUANTS ===========================================================
    //[https://github.com/ggml-org/llama.cpp/pull/4773]
    IQ4_NL,
    IQ4_XS,
    IQ3_S,
    IQ3_XXS,
    IQ2_S,
    IQ2_XS,
    IQ2_XXS,
    IQ1_M,
    IQ1_S,

    //== LEGACY QUANTS ======================================================
    Q8_1,
    Q8_0,
    Q5_1,
    Q5_0,
    Q4_1,
    Q4_0
};

/**
 * Converts a C++ data type to its corresponding DType enum value.
 */
template <typename T>
constexpr tin::DType get_dtype() {
    if      constexpr (std::is_same_v<T, double>) return tin::DType::F64;
    else if constexpr (std::is_same_v<T, float> ) return tin::DType::F32;
    else if constexpr (std::is_same_v<T, int>   ) return tin::DType::I32;
    else if constexpr (std::is_same_v<T, int8_t>) return tin::DType::I8;
    else if constexpr (std::is_same_v<T, bool>  ) return tin::DType::BOOL;
    else static_assert(false, "Unsupported type for DType conversion");
}

} // namespace tin

namespace std { string_view to_string(::tin::DType type); }

/**
 * Overloads the insertion (<<) operator for outputting tin::DType values to streams.
 *
 * This inline function allows objects of type `tin::DType` to be directly
 * inserted into output streams such as `std::cout`.
 *
 * @param os   The output stream where the thin::DType will be written.
 * @param type The tin::DType value to be inserted into the output stream.
 * @return A reference to 'os' for chaining.
 */
inline 
std::ostream& operator<<(std::ostream& os, tin::DType type) {
    return os << std::to_string(type);
}

/**
 * Formatter specialization for DType type.
 * 
 * Enables using C++20's formatting framework with tin::DType valuess. The
 * formatter converts the DType value to its string representation and
 * formats it accordingly.
 * 
 * @code
 * #include <iostream>
 * #include <format>
 * 
 * int main() {
 *     auto myType = DType::F16;
 *     std::cout << std::format("The type is {}", myType) << std::endl;
 *     // output = "The type is F16"
 * }
 * @endcode
 */
template <>
struct std::formatter<tin::DType> : std::formatter<std::string_view>
{
    auto format(const tin::DType& dtype, format_context& ctx) const {
        return std::formatter<std::string_view>::format(std::to_string(dtype), ctx);
    }
};


#endif // TIN_DTYPE_H_
