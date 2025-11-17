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
#include <iostream>           // for std::istream, std::cout, std::endl
#include <tin/tensormap.h>    // for TensorMap
#include "gguf/ggufistream.h" // for GGUFIStream
namespace tin {
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::nullopt;

// The maximum length of a key in the metadata section.
static const size_t MaximumKeyLength = 65536;



// inserta un key-value pair from the metadata section into a TensorMap.
static ReadError
_insert_metadata(TensorMap& tensorMap, GGUFIStream& istream, char* keyBuffer, size_t keyBufferSize)
{
    // read the key length
    auto optionalKeyLength = istream.read_le_uint64();
    if( !optionalKeyLength ) { return ReadError::MissingData; }
    auto keyLength = optionalKeyLength.value();

    // if length is out of bounds, return with error
    if( keyLength<=0 || keyLength>keyBufferSize ) { return ReadError::InvalidFormat; }
    if( !istream.read(keyBuffer, keyLength)     ) { return ReadError::MissingData;   }

    auto key = StringView(keyBuffer, keyLength);

    // read the value type
    auto metadataValueType = istream.read_metadata_value_type();
    if( !metadataValueType ) { return ReadError::MissingData; }
    switch( *metadataValueType )
    {
        // 1-byte value where 0 is false and 1 is true.
        // anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
        case GGUFMetadataValueType::BOOL: {
            auto value = istream.read_uint8();
            if( !value )      { return ReadError::MissingData;   }
            if( *value >= 2 ) { return ReadError::InvalidFormat; }
            tensorMap.metadata().set_boolean(key, *value!=0, StorageType::UINT8);
            } break;

        case GGUFMetadataValueType::INT8: {
            auto value = istream.read_int8();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_integer(key, *value, StorageType::INT8);
            } break;

        case GGUFMetadataValueType::UINT8: {
            auto value = istream.read_uint8();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_integer(key, *value, StorageType::UINT8);
            } break;

        case GGUFMetadataValueType::INT16: {
            auto value = istream.read_le_int16();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_integer(key, *value, StorageType::INT16);
            } break;

        case GGUFMetadataValueType::UINT16: {
            auto value = istream.read_le_uint16();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_integer(key, *value, StorageType::UINT16);
            } break;

        case GGUFMetadataValueType::INT32: {
            auto value = istream.read_le_int32();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_integer(key, *value, StorageType::INT32);
            } break;

        case GGUFMetadataValueType::UINT32: {
            auto value = istream.read_le_uint32();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_integer(key, *value, StorageType::UINT32);
            } break;

        case GGUFMetadataValueType::INT64: {
            auto value = istream.read_le_int64();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_integer(key, *value, StorageType::INT64);
            } break;

        case GGUFMetadataValueType::UINT64: {
            auto value = istream.read_le_uint64();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_integer(key, *value, StorageType::UINT64);
            } break;

        case GGUFMetadataValueType::FLOAT32: {
            auto value = istream.read_float32();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_float(key, *value, StorageType::FLOAT32);
            } break;

        case GGUFMetadataValueType::FLOAT64: {
            auto value = istream.read_float64();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_float(key, *value, StorageType::FLOAT64);
            } break;

        case GGUFMetadataValueType::STRING: {
            auto value = istream.read_string();
            if( !value ) { return ReadError::MissingData; }
            tensorMap.metadata().set_string(key, *value);
            } break;

        case GGUFMetadataValueType::ARRAY:
            std::cout << "## ignoring: " << key << std::endl;
            if( !istream.ignore_array() ) { return ReadError::MissingData; }
            break;
    }
    return ReadError::None;
}

//======================== READING FROM GGUF FILE =========================//

TensorMap
TensorMap::_fromgguf(const uint8_t   firstBytes[8],
                     std::istream&   istream_,
                     ReadError&      outError,
                     const Path&     filePath,          // = {},
                     std::streamsize fileSize,          // = 0,
                     std::streampos  byteBufferPosition // = 0
) noexcept {
    TensorMap   tensorMap;
    GGUFIStream istream{ istream_ } ;

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
    if( ggufIdentifier != 0x46554747 ) {
        outError = ReadError::InvalidFormat;
        return tensorMap;
    }
    if( ggufVersion != 3 ) {
        outError = ReadError::UnsupportedVersion;
        return tensorMap;
    }
    const auto tensorCount     = istream.read_le_uint64();
    const auto metadataKvCount = istream.read_le_uint64();
    if( !tensorCount || !metadataKvCount ) { outError = ReadError::InvalidFormat; return tensorMap; }

    // DEBUG
    std::cout << "## Number of tensors:  " << *tensorCount     << std::endl;
    std::cout << "## Number of metadata: " << *metadataKvCount << std::endl;

    auto keyBuffer = std::make_unique<char[]>(MaximumKeyLength);
    for( int i=0 ; i<metadataKvCount ; ++i ) {
        if( outError != ReadError::None ) { break; }
        outError = _insert_metadata(tensorMap, istream, keyBuffer.get(), MaximumKeyLength);
    }

    return tensorMap;
}


//========================= WRITING TO GGUF FILE ==========================//





} // namespace tin