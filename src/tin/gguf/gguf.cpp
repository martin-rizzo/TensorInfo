/*
| File    : gguf/gguf.cpp
| Purpose : Helper functions and constants for reading/writing GGUF files
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 11, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <bit>     // for std::endian
#include <cstdint> // for std::uint*_t
#include <limits>  // for std::numeric_limits
#include <iostream> // for std::cout, std::endl
#include "gguf.h"
namespace tin {
using std::nullopt;

/**
 * Swaps the byte order of an unsigned integral type.
 *
 * This function performs a byte swap operation on the given unsigned integral
 * value. It supports 16-bit, 32-bit, and 64-bit integers by reversing their 
 * byte order (endianness).
 *
 * @tparam T The type of the unsigned integral value.
 *           Must be one of std::uint16_t, std::uint32_t, or std::uint64_t.
 * 
 * @param value The unsigned integer whose byte order is to be swapped.
 * @return The byte-swapped version of the input value.
 */
template<std::unsigned_integral T>
constexpr T byteswap(T value) noexcept {
    if constexpr( sizeof(T) == 2 ) { // (2 bytes)
        return static_cast<T>(
            ((static_cast<uint16_t>(value) >> 8) & 0x00FF) | 
            ((static_cast<uint16_t>(value) << 8) & 0xFF00)
        );
    } else if constexpr (sizeof(T) == 4) { // (4 bytes)
        return static_cast<T>(
            ((static_cast<uint32_t>(value) >> 24) & 0x000000FFUL) |
            ((static_cast<uint32_t>(value) >> 8)  & 0x0000FF00UL) |
            ((static_cast<uint32_t>(value) << 8)  & 0x00FF0000UL) |
            ((static_cast<uint32_t>(value) << 24) & 0xFF000000UL)
        );
    } else if constexpr (sizeof(T) == 8) { // (8 bytes)
        return static_cast<T>(
            ((static_cast<uint64_t>(value) >> 56) & 0x00000000000000FFULL) |
            ((static_cast<uint64_t>(value) >> 40) & 0x000000000000FF00ULL) |
            ((static_cast<uint64_t>(value) >> 24) & 0x0000000000FF0000ULL) |
            ((static_cast<uint64_t>(value) >> 8)  & 0x00000000FF000000ULL) |
            ((static_cast<uint64_t>(value) << 8)  & 0x000000FF00000000ULL) |
            ((static_cast<uint64_t>(value) << 24) & 0x0000FF0000000000ULL) |
            ((static_cast<uint64_t>(value) << 40) & 0x00FF000000000000ULL) |
            ((static_cast<uint64_t>(value) << 56) & 0xFF00000000000000ULL)
        );
    } else {
        static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, 
                      "byteswap only supports integers of 16, 32 or 64 bits.");
    }
}

template<typename T>
Optional<T> read_le(std::istream& istream) noexcept {
    T value;
    if( !istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return nullopt; }
    if constexpr ( std::endian::native == std::endian::big ) { value = byteswap(value); }
    return value;
}

template<typename T>
Optional<T> read_be(std::istream& istream) noexcept {
    T value;
    if( !istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return nullopt; }
    if constexpr ( std::endian::native == std::endian::little ) { value = byteswap(value); }
    return value;
}

//=========================== READING INTEGERS ============================//

Optional<int>
GGUF::read_int8(std::istream& istream) noexcept {
    std::int8_t value;
    if( !istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return nullopt; }
    return int{ value };
}

Optional<unsigned>
GGUF::read_uint8(std::istream& istream) noexcept {
    std::uint8_t value;
    if( !istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return nullopt; }
    return unsigned{ value };
}

Optional<int>
GGUF::read_le_int16(std::istream& istream) noexcept {
    const auto value = read_le<std::int16_t>(istream);
    return value ? Optional<int>{*value} : nullopt;
}

Optional<unsigned>
GGUF::read_le_uint16(std::istream& istream) noexcept {
    const auto value = read_le<std::uint16_t>(istream);
    return value ? Optional<unsigned>{*value} : nullopt;
}

Optional<int>
GGUF::read_le_int32(std::istream& istream) noexcept {
    const auto value = read_le<std::uint32_t>(istream);
    return value ? Optional<int>{*value} : nullopt;
}

Optional<unsigned>
GGUF::read_le_uint32(std::istream& istream) noexcept {
    const auto value = read_le<std::uint32_t>(istream);
    return value ? Optional<unsigned>{*value} : nullopt;
}

Optional<long>
GGUF::read_le_int64(std::istream& istream) noexcept {
    const auto value = read_le<std::uint64_t>(istream);
    return value ? Optional<long>{*value} : nullopt;
}

Optional<unsigned long>
GGUF::read_le_uint64(std::istream& istream) noexcept {
    const auto value = read_le<std::uint64_t>(istream);
    return value ? Optional<unsigned long>{*value} : nullopt;
}

//====================== READING FLOAT-POINT NUMBERS ======================//


Optional<float>
GGUF::read_float32(std::istream& istream) noexcept {
    static_assert( std::numeric_limits<float>::is_iec559 == true, "IEEE 754 is required for GGUF FLOAT32" );
    static_assert( std::numeric_limits<float>::radix     ==  2  , "'float' must be base-2 for GGUF FLOAT32 (IEEE 754)" );
    static_assert( std::numeric_limits<float>::digits    == 24  , "'float' must have 24 bits of precision for GGUF FLOAT32 (IEEE 754)" );
    static_assert(         sizeof(float)  *  CHAR_BIT    == 32  , "'float' must be 32 bits, it is required for GGUF FLOAT32" );
    
    float value;
    if( !istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return nullopt; }
    return value;
}

Optional<double>
GGUF::read_float64(std::istream& istream) noexcept {
    static_assert( std::numeric_limits<double>::is_iec559 == true, "IEEE 754 is required for GGUF FLOAT64" );
    static_assert( std::numeric_limits<double>::radix     ==  2  , "'double' must be base-2 for GGUF FLOAT64 (IEEE 754)");
    static_assert( std::numeric_limits<double>::digits    == 53  , "'double' must have 53 bits of precision for GGUF FLOAT64 (IEEE 754)" );
    static_assert(         sizeof(double)  *  CHAR_BIT    == 64  , "'double' must be 64 bits, it is required for GGUF FLOAT64" );

    double value;
    if( !istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return nullopt; }
    return value;
}


//========================== READING OTHER VALUES ==========================//

Optional<String>
GGUF::read_string(std::istream& istream) noexcept {
    auto length = read_le_uint64(istream);
    if( !length ) { return nullopt; }
    auto string = String( *length, '\0' );
    if( !istream.read(string.data(), string.size()) ) { return nullopt; }
    return string;
}


} // namespace tin
