/*
| File    : gguf/ggufisream.h
| Purpose : Wrapper that provides convenient methods for reading GGUF files.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 16, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_GGUFISREAM_H_
#define TIN_GGUFISREAM_H_
#include "ggufconstants.h" // for GGUFMetadataValueType
#include <tin/common.h> // for Optional<T>, String, ULong, Long
namespace tin {


/**
 * Wrapper for `std::istream` that provides convenient methods for reading GGUF files.
 * 
 * This class encapsulates an istream reference and provides methods to read
 * various GGUF data types in a safe manner, returning optional values to
 * indicate success or failure of operations.
 */
class GGUFIStream
{
// CONSTRUCTION/DESTRUCTION
public:
    explicit GGUFIStream(std::istream& istream) noexcept;
    ~GGUFIStream() = default;

// READING GGUF DATA TYPES
public:
    bool                read(char* buffer, std::streamsize size) noexcept;
    Optional<int>       read_int8()       noexcept;
    Optional<unsigned>  read_uint8()      noexcept;
    Optional<int>       read_le_int16()   noexcept;
    Optional<unsigned>  read_le_uint16()  noexcept;
    Optional<int>       read_le_int32()   noexcept;
    Optional<unsigned>  read_le_uint32()  noexcept;
    Optional<Long>      read_le_int64()   noexcept;
    Optional<ULong>     read_le_uint64()  noexcept;
    Optional<float>     read_float32()    noexcept;
    Optional<double>    read_float64()    noexcept;
    Optional<String>    read_string()     noexcept;
    Optional<GGUFMetadataValueType> read_metadata_value_type() noexcept;

// IGNORING GGUF DATA TYPES
public:
    bool ignore_string() noexcept;
    bool ignore_array()  noexcept;

// STATE FLAGS
public:
    void clear() noexcept;
    bool good() const noexcept;
    bool fail() const noexcept;
    bool eof() const noexcept;

// OPERATOR OVERLOADS
public:
    operator bool()   const noexcept { return !_istream.fail(); }
    bool operator !() const noexcept { return  _istream.fail(); }

    
// IMPLEMENTATION
private:
    std::istream& _istream; //!< Reference to the underlying input stream
};


//========================= INLINES: STATE FLAGS ==========================//

/**
 * Clears any error flags of the stream.
 *
 * This function resets all error conditions on the internal input stream,
 * allowing subsequent I/O operations to proceed without being affected by
 * previous errors. It essentially makes the stream "good" again.
 */
inline void
GGUFIStream::clear() noexcept { _istream.clear(); }

/**
 * Checks if the stream is in a valid state for I/O operations.
 *
 * This function returns `true` if none of the error flags (failbit, badbit)
 * are set on the internal input stream. It indicates that the stream is
 * ready and capable of performing input/output operations without any errors.
 *
 * @return `true` if the stream is in a good state for I/O; otherwise, `false`.
 */
inline bool
GGUFIStream::good() const noexcept { return _istream.good(); }

/**
 * Checks if an error has occurred on the stream during the last operation.
 *
 * This function returns `true` if either the failbit or badbit is set on
 * the internal input stream. It signifies that a logical error (failbit) or
 * a more serious issue such as loss of integrity (badbit) has been encountered.
 *
 * @return `true` if an error occurred during the last I/O operation; otherwise, `false`.
 */
inline bool
GGUFIStream::fail() const noexcept { return _istream.fail(); }

/**
 * Checks if the end-of-file condition has been met on the stream.
 *
 * This function returns `true` if the end-of-file (EOF) flag is set for
 * the internal input stream. It indicates that all available input operations 
 * have been exhausted, typically meaning no more data can be read from the source.
 *
 * @return `true` if the EOF condition has been reached; otherwise, `false`.
 */
inline bool
GGUFIStream::eof() const noexcept  { return _istream.eof(); }


}      // namespace tin
#endif // TIN_GGUFISREAM_H_