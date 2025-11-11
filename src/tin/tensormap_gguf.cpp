/*
| File    : tensormap_gguf.cpp
| Purpose : TensorMap GGUF file reader/writer methods.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 9, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|
|  GGUF Format Specification: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
|  GGUF Format HuggingFace  : https://huggingface.co/docs/hub/en/gguf
|
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <iostream>        // for std::istream, std::cout, std::endl
#include <tin/tensormap.h> // for TensorMap
namespace tin {
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;

//================================ HELPERS ================================//

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

#if 0
static std::optional<uint32_t>
_read_le_uint32(std::istream& istream) noexcept {
    uint32_t value;
    if( !istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return std::nullopt; }
    if constexpr ( std::endian::native == std::endian::big ) { value = byteswap(value); }
    return value;
}
#endif

/**
 * Reads a little-endian uint64_t from an input stream.
 *
 * Attempts to read a 64-bit unsigned integer in little-endian format from
 * the given input stream. If reading fails, it returns an empty optional.
 * @param istream A reference to an input stream from which data is read.
 * @return An optional containing the uint64_t read from the stream,
 *         or an empty optional if reading fails.
 */
static std::optional<uint64_t>
_read_le_uint64(std::istream& istream) noexcept {
    uint64_t value;
    if( !istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return std::nullopt; }
    if constexpr ( std::endian::native == std::endian::big ) { value = byteswap(value); }
    return value;
}

//======================== READING FROM GGUF FILE =========================//

TensorMap
TensorMap::_fromgguf(const uint8_t   firstBytes[8],
                     std::istream&   istream,
                     ReadError&      outError,
                     const Path&     filePath,          // = {},
                     std::streamsize fileSize,          // = 0,
                     std::streampos  byteBufferPosition // = 0
) noexcept {
    TensorMap tensorMap;

    // first 8 bytes are gguf-identifier and version number
    const unsigned ggufIdentifier{ (static_cast<unsigned>(firstBytes[0]) << 0) |
                                   (static_cast<unsigned>(firstBytes[1]) << 8) |
                                   (static_cast<unsigned>(firstBytes[2]) <<16) |
                                   (static_cast<unsigned>(firstBytes[3]) <<24)
                                 };
    const unsigned ggufVersion   { (static_cast<unsigned>(firstBytes[4]) << 0) |
                                   (static_cast<unsigned>(firstBytes[5]) << 8) |
                                   (static_cast<unsigned>(firstBytes[6]) <<16) |
                                   (static_cast<unsigned>(firstBytes[7]) <<24)
                                 };
    if( ggufIdentifier != 0x46554747 ) { return tensorMap; /*"Invalid magic number."*/ }
    if( ggufVersion    != 3 )          { return tensorMap; /*"Unsupported version."*/  }

    const auto tensorCount     = _read_le_uint64(istream);
    const auto metadataKvCount = _read_le_uint64(istream);
    if( !tensorCount || !metadataKvCount ) { outError = ReadError::InvalidFormat; return tensorMap; }

    std::cout << "## Number of tensors:  " << *tensorCount     << std::endl;
    std::cout << "## Number of metadata: " << *metadataKvCount << std::endl;

    return tensorMap;
}


//========================= WRITING TO GGUF FILE ==========================//





} // namespace tin