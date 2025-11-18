/*
| File    : gguf/ggufisream.cpp
| Purpose : Wrapper that provides convenient methods for reading GGUF files.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 16, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include "ggufistream.h"
namespace tin {
using std::nullopt;

//======================= CONSTRUCTION/DESTRUCTION ========================//

/**
 * Constructs a GGUFIStream associating it with an existing `std::istream`.
 * @param istream        A reference to the input stream that this GGUFIStream will use.
 * @param maxStringSize  The maximum size (in bytes) of a string to read from the stream,
 *                       used to prevent memory exhaustion attacks. Any strings longer
 *                       than this value will be truncated. A value of 0 means no limits.
 *                       Defaults to 256KB.
 */
GGUFIStream::GGUFIStream(std::istream& istream,
                         ULong         maxStringSize // = 256 * 1024
) noexcept
: _istream{istream}, _maxStringSize{maxStringSize}
{}

//======================== READING GGUF DATA TYPES ========================//

/**
 * Reads data from the associated input stream into a buffer.
 *
 * This function reads a specified number of bytes from the underlying
 * `std::istream` into the provided buffer. It returns a boolean indicating
 * whether the read operation was successful.
 *
 * @param buffer A pointer to the memory location where read data will be stored.
 * @param size   The number of bytes to read from the stream.
 * @return `true` if the read operation was successful, `false` otherwise.
 */
bool
GGUFIStream::read(char* buffer, std::streamsize size) noexcept {
    return bool{ _istream.read(buffer, size) };
}

/**
 * Reads an 8-bit signed integer from the stream.
 *
 * This function attempts to read one byte from the input stream, interpreting
 * it as a signed integer. If the read operation fails or encounters an error,
 * it returns an empty optional to indicate failure.
 *
 * @return An optional containing the read signed integer if successful, or
 *         an empty optional if an error occurs during reading.
 */
Optional<int>
GGUFIStream::read_int8() noexcept {
    char buffer[1];
    if( !_istream.read(buffer, sizeof(buffer)) ) { return nullopt; }
    return
        static_cast<signed char>(buffer[0]);
}

/**
 * Reads an 8-bit unsigned integer from the stream.
 *
 * This function reads one byte from the input stream and interprets it as
 * an unsigned integer. If the read operation fails or encounters an error,
 * it returns an empty optional to indicate failure.
 *
 * @return An optional containing the read unsigned integer if successful, or
 *         an empty optional if an error occurs during reading.
 */
Optional<unsigned>
GGUFIStream::read_uint8() noexcept {
    char buffer[1];
    if( !_istream.read(buffer, sizeof(buffer)) ) { return nullopt; }
    return
        static_cast<unsigned char>(buffer[0]);
}

/**
 * Reads a 16-bit little-endian signed integer from the stream.
 *
 * This function reads two bytes from the input stream and interprets them
 * as a single 16-bit signed integer in little-endian format. If the read
 * operation fails or encounters an error, it returns an empty optional to 
 * indicate failure.
 *
 * @return An optional containing the read signed integer if successful, or
 *         an empty optional if an error occurs during reading.
 */
Optional<int>
GGUFIStream::read_le_int16() noexcept {
    char buffer[2];
    if( !_istream.read(buffer, sizeof(buffer)) ) { return nullopt; }
    return
        static_cast<unsigned char>(buffer[0]) << 0 |
        static_cast<signed   char>(buffer[1]) << 8;
}

/**
 * Reads a 16-bit little-endian unsigned integer from the stream.
 *
 * This function reads two bytes from the input stream and interprets them
 * as a single 16-bit unsigned integer in little-endian format. If the read 
 * operation fails or encounters an error, it returns an empty optional to 
 * indicate failure.
 *
 * @return An optional containing the read unsigned integer if successful, or
 *         an empty optional if an error occurs during reading.
 */
Optional<unsigned>
GGUFIStream::read_le_uint16() noexcept {
    char buffer[2];
    if( !_istream.read(buffer, sizeof(buffer)) ) { return nullopt; }
    return static_cast<unsigned>(
        static_cast<unsigned char>(buffer[0]) << 0 |
        static_cast<unsigned char>(buffer[1]) << 8
    );
}

/**
 * Reads a 32-bit little-endian signed integer from the stream.
 *
 * This function reads four bytes from the input stream and interprets them
 * as a single 32-bit signed integer in little-endian format. If the read 
 * operation fails or encounters an error, it returns an empty optional to 
 * indicate failure.
 *
 * @return An optional containing the read signed integer if successful, or
 *         an empty optional if an error occurs during reading.
 */
Optional<int>
GGUFIStream::read_le_int32() noexcept {
    char buffer[4];
    if( !_istream.read(buffer, sizeof(buffer)) ) { return nullopt; }
    return
        static_cast<unsigned char>(buffer[0]) <<  0 |
        static_cast<unsigned char>(buffer[1]) <<  8 |
        static_cast<unsigned char>(buffer[2]) << 16 | 
        static_cast<signed   char>(buffer[3]) << 24;
}

/**
 * Reads a 32-bit little-endian unsigned integer from the stream.
 *
 * This function reads four bytes from the input stream and interprets them
 * as a single 32-bit unsigned integer in little-endian format. If the read 
 * operation fails or encounters an error, it returns an empty optional to 
 * indicate failure.
 *
 * @return An optional containing the read unsigned integer if successful, or
 *         an empty optional if an error occurs during reading.
 */
Optional<unsigned>
GGUFIStream::read_le_uint32() noexcept {
    char buffer[4];
    if( !_istream.read(buffer, sizeof(buffer)) ) { return nullopt; }
    return static_cast<unsigned>(
        static_cast<unsigned char>(buffer[0]) <<  0 |
        static_cast<unsigned char>(buffer[1]) <<  8 |
        static_cast<unsigned char>(buffer[2]) << 16 | 
        static_cast<unsigned char>(buffer[3]) << 24
    );
}


/**
 * Reads a 64-bit little-endian signed integer from the stream.
 *
 * This function reads eight bytes from the input stream and interprets them
 * as a single 64-bit signed integer in little-endian format. If the read 
 * operation fails or encounters an error, it returns an empty optional to 
 * indicate failure.
 *
 * @return An optional containing the read signed long integer if successful,
 *         or an empty optional if an error occurs during reading.
 */
Optional<Long>
GGUFIStream::read_le_int64() noexcept {
    char buffer[8];
    if( !_istream.read(buffer, sizeof(buffer)) ) { return nullopt; }
    return
        static_cast<Long>(static_cast<unsigned char>(buffer[0])) <<  0 |
        static_cast<Long>(static_cast<unsigned char>(buffer[1])) <<  8 |
        static_cast<Long>(static_cast<unsigned char>(buffer[2])) << 16 | 
        static_cast<Long>(static_cast<unsigned char>(buffer[3])) << 24 |
        static_cast<Long>(static_cast<unsigned char>(buffer[4])) << 32 |
        static_cast<Long>(static_cast<unsigned char>(buffer[5])) << 40 |
        static_cast<Long>(static_cast<unsigned char>(buffer[6])) << 48 |
        static_cast<Long>(static_cast<signed   char>(buffer[7])) << 56;
}

/**
 * Reads a 64-bit little-endian unsigned integer from the stream.
 *
 * This function reads eight bytes from the input stream and interprets them
 * as a single 64-bit unsigned integer in little-endian format. If the read 
 * operation fails or encounters an error, it returns an empty optional to 
 * indicate failure.
 *
 * @return An optional containing the read unsigned long integer if successful,
 *         or an empty optional if an error occurs during reading.
 */
Optional<ULong>
GGUFIStream::read_le_uint64() noexcept {
    char buffer[8];
    if( !_istream.read(buffer, sizeof(buffer)) ) { return nullopt; }
    return
        static_cast<ULong>(static_cast<unsigned char>(buffer[0])) <<  0 |
        static_cast<ULong>(static_cast<unsigned char>(buffer[1])) <<  8 |
        static_cast<ULong>(static_cast<unsigned char>(buffer[2])) << 16 | 
        static_cast<ULong>(static_cast<unsigned char>(buffer[3])) << 24 |
        static_cast<ULong>(static_cast<unsigned char>(buffer[4])) << 32 |
        static_cast<ULong>(static_cast<unsigned char>(buffer[5])) << 40 |
        static_cast<ULong>(static_cast<unsigned char>(buffer[6])) << 48 |
        static_cast<ULong>(static_cast<unsigned char>(buffer[7])) << 56;
}

/**
 * Reads a 32-bit floating-point number (float) from the stream.
 *
 * This function reads four bytes from the input stream and interprets them as 
 * a single 32-bit floating-point number in IEEE 754 format. If the read operation 
 * fails or encounters an error, it returns an empty optional to indicate failure.
 *
 * @return An optional containing the read float if successful, or
 *         an empty optional if an error occurs during reading.
 */
Optional<float>
GGUFIStream::read_float32() noexcept {
    static_assert( std::numeric_limits<float>::is_iec559 == true, "IEEE 754 is required for GGUF FLOAT32" );
    static_assert( std::numeric_limits<float>::radix     ==  2  , "'float' must be base-2 for GGUF FLOAT32 (IEEE 754)" );
    static_assert( std::numeric_limits<float>::digits    == 24  , "'float' must have 24 bits of precision for GGUF FLOAT32 (IEEE 754)" );
    static_assert(         sizeof(float)  *  CHAR_BIT    == 32  , "'float' must be 32 bits, it is required for GGUF FLOAT32" );
    
    float value;
    if( !_istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return nullopt; }
    return value;
}

/**
 * Reads a 64-bit floating-point number (double) from the stream.
 *
 * This function reads eight bytes from the input stream and interprets them as 
 * a single 64-bit floating-point number in IEEE 754 format. If the read operation 
 * fails or encounters an error, it returns an empty optional to indicate failure.
 *
 * @return An optional containing the read double if successful, or
 *         an empty optional if an error occurs during reading.
 */

Optional<double>
GGUFIStream::read_float64() noexcept {
    static_assert( std::numeric_limits<double>::is_iec559 == true, "IEEE 754 is required for GGUF FLOAT64" );
    static_assert( std::numeric_limits<double>::radix     ==  2  , "'double' must be base-2 for GGUF FLOAT64 (IEEE 754)");
    static_assert( std::numeric_limits<double>::digits    == 53  , "'double' must have 53 bits of precision for GGUF FLOAT64 (IEEE 754)" );
    static_assert(         sizeof(double)  *  CHAR_BIT    == 64  , "'double' must be 64 bits, it is required for GGUF FLOAT64" );

    double value;
    if( !_istream.read(reinterpret_cast<char*>(&value), sizeof(value)) ) { return nullopt; }
    return value;
}

/**
 * Reads a GGUF string from the stream.
 *
 * @return An optional containing the read string if successful, or
 *         an empty optional if an error occurs during reading.
 */
Optional<String>
GGUFIStream::read_string() noexcept {
    auto length = read_le_uint64();
    if( !length ) { return nullopt; }

    // read at most `_maxStringLength` characters,
    // ignoring any additional characters
    ULong stringSize = *length;
    ULong ignoreSize = 0;
    if( _maxStringSize>0 && stringSize > _maxStringSize ) {
        ignoreSize =  stringSize - _maxStringSize;
        stringSize = _maxStringSize;
    }
    String string( stringSize, '\0' );
    if( !_istream.read(string.data(), string.size())      ) { return nullopt; }
    if( ignoreSize>0 && !_istream.ignore(ignoreSize ) ) { return nullopt; }
    return string;
}

Optional<GGMLType>
GGUFIStream::read_ggml_type() noexcept {
    auto uint32 = read_le_uint32();
    if( !uint32 ) { return nullopt; }
    return static_cast<GGMLType>(*uint32);
}

/**
 * Reads a metadata value type from the stream.
 *
 * @return An optional containing the `GGUFMetadataValueType` if successful, or
 *         an empty optional if an error occurs during reading.
 */
Optional<GGUFMetadataValueType>
GGUFIStream::read_metadata_value_type() noexcept {
    const auto type = read_le_uint32();
    if( !type ) { return nullopt; }
    return static_cast<GGUFMetadataValueType>( *type );
}

// lee una secuencia de enteros uint64_t de longitud determinada
// y lo sretorna como un vector de ULong
Optional<Vector<ULong>>
GGUFIStream::read_seq_of_le_uint64(unsigned size) noexcept {
    Vector<ULong> sequence;
    sequence.reserve( size );
    for( unsigned i=0 ; i<size ; ++i ) {
        auto value = read_le_uint64();
        if( !value ) { return nullopt; }
        sequence.push_back(*value);
    }
    return sequence;
}

//======================= IGNORING GGUF DATA TYPES ========================//

/**
 * Ignores a GGUF string from the stream.
 *
 * This function skips over a GGUF string by ignoring all its characters.
 * It returns `true` if the string was successfully skipped, and `false`
 * if an error occurs during the process.
 *
 * @return `true` if the string was successfully ignored, or
 *         `false` if an error occurs during the process.
 */
bool
GGUFIStream::ignore_string() noexcept {
    const auto length = read_le_uint64();
    if( !length ) { return false; }
    _istream.ignore(*length);
    return true;
}

/**
 * Ignores a GGUF array from the stream.
 *
 * This function skips over a GGUF array by ignoring each of its elements.
 * The istream is expected to be positioned at the beginning of the array
 * structure, and when the function returns, it will be positioned at the
 * beginning of the next object after the array.
 *
 * @return `true` if the array was successfully ignored without errors, or
 *         `false` if an error occurs during the process.
 */
bool
GGUFIStream::ignore_array() noexcept {
    const auto metadataValueType = read_metadata_value_type();
    if( !metadataValueType ) { return false; }

    const auto arrayLength = read_le_uint64();
    if( !arrayLength ) { return false; }

    // debug message
    // std::cout << "## skipping array of " << to_string(*metadataValueType) << "[" << *arrayLength << "]" << std::endl;

    switch( *metadataValueType )
    {
        case GGUFMetadataValueType::BOOL:
        case GGUFMetadataValueType::INT8:
        case GGUFMetadataValueType::UINT8:
            _istream.ignore( *arrayLength );
            break;

        case GGUFMetadataValueType::INT16:
        case GGUFMetadataValueType::UINT16:
            _istream.ignore( *arrayLength * 2 );
            break;

        case GGUFMetadataValueType::INT32:
        case GGUFMetadataValueType::UINT32:
        case GGUFMetadataValueType::FLOAT32:
            _istream.ignore( *arrayLength * 4 );
            break;

        case GGUFMetadataValueType::INT64:
        case GGUFMetadataValueType::UINT64:
        case GGUFMetadataValueType::FLOAT64:
            _istream.ignore( *arrayLength * 8 );
            break;

        case GGUFMetadataValueType::STRING:
            for( auto c = *arrayLength ; c ; --c ) { ignore_string(); }
            break;

        case GGUFMetadataValueType::ARRAY:
            for( auto c = *arrayLength ; c ; --c ) { ignore_array(); }
            break;
    }
    return _istream.good();
}




} // namespace tin
