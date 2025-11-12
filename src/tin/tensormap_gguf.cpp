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
#include "gguf/gguf.h"     // helper functions & constants GGUF file I/O
namespace tin {
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::nullopt;

// The maximum length of a key in the metadata section.
static const size_t MaximumKeyLength = 65536;



static ReadError
_add_metadata(TensorMap& tensorMap, std::istream& istream, char* keyBuffer, size_t keyBufferSize)
{
    auto optionalKeyLength = GGUF::read_le_uint64(istream);
    if( !optionalKeyLength ) { return ReadError::FileTruncated; }
    auto keyLength = optionalKeyLength.value();

    // if length is out of bounds, return with error
    if( keyLength<=0 || keyBufferSize<keyLength ) { return ReadError::InvalidFormat; }
    if( !istream.read(keyBuffer, keyLength)     ) { return ReadError::FileTruncated; }

    auto key = StringView(keyBuffer, keyLength);

    auto optionalValueType = GGUF::read_le_int32(istream);
    if( !optionalValueType ) { return ReadError::FileTruncated; }
    auto valueType = GGUF::METADATA_VALUE_TYPE{ optionalValueType.value() };

    switch( valueType ) {
        case GGUF::METADATA_VALUE_TYPE::BOOL:
            std::cout << "tensorMap.metadata().add_bool";
            break;

        case GGUF::METADATA_VALUE_TYPE::UINT8: {
            auto value = GGUF::read_uint8(istream);
            if( !value ) { return ReadError::FileTruncated; }
            std::cout << "tensorMap.metadata().add_uint(" << key << ", " << (*value) << ")";
            } break;

        case GGUF::METADATA_VALUE_TYPE::INT8: {
            auto value = GGUF::read_int8(istream);
            if( !value ) { return ReadError::FileTruncated; }
            std::cout << "tensorMap.metadata().add_int(" << key << ", " << (*value) << ")";
            } break;

        case GGUF::METADATA_VALUE_TYPE::UINT16: {
            auto value = GGUF::read_le_uint16(istream);
            if( !value ) { return ReadError::FileTruncated; }
            std::cout << "tensorMap.metadata().add_uint'"  << key << "', " << (*value) << ", UINT16)" << std::endl;
            } break;

        case GGUF::METADATA_VALUE_TYPE::INT16:
            std::cout << "tensorMap.metadata().add_int";
            break;
        case GGUF::METADATA_VALUE_TYPE::UINT32: {
            auto value = GGUF::read_le_uint32(istream);
            if( !value ) { return ReadError::FileTruncated; }
            std::cout << "tensorMap.metadata().add_uint('" << key << "', " << (*value) << ", UINT32)" << std::endl;
            } break;

        case GGUF::METADATA_VALUE_TYPE::INT32:
            std::cout << "tensorMap.metadata().add_int";
            break;

        case GGUF::METADATA_VALUE_TYPE::UINT64: {
            auto value = GGUF::read_le_uint64(istream);
            if( !value ) { return ReadError::FileTruncated; }
            std::cout << "tensorMap.metadata().add_uint('" << key << "', " << (*value) << ", UINT64)" << std::endl;
            } break;

            case GGUF::METADATA_VALUE_TYPE::INT64:
            std::cout << "tensorMap.metadata().add_int";
            break;

        case GGUF::METADATA_VALUE_TYPE::FLOAT32: {
            auto value = GGUF::read_float32(istream);
            if( !value ) { return ReadError::FileTruncated; }
            std::cout << "tensorMap.metadata().add_float('" << key << "', " << (*value) << ", FLOAT32)" << std::endl;
            } break;

        case GGUF::METADATA_VALUE_TYPE::FLOAT64: {
            auto value = GGUF::read_float64(istream);
            if( !value ) { return ReadError::FileTruncated; }
            std::cout << "tensorMap.metadata().add_double('" << key << "', " << (*value) << ", FLOAT64)" << std::endl;
            } break;

        case GGUF::METADATA_VALUE_TYPE::STRING: {
            auto value = GGUF::read_string(istream);
            if( !value ) { return ReadError::FileTruncated; }
            std::cout << "tensorMap.metadata().add_string('" << key << "', '" << (*value) << "')" << std::endl;
            } break;

        case GGUF::METADATA_VALUE_TYPE::ARRAY:
            std::cout << "tensorMap.metadata().add_string_array";
            break;
    }

    return ReadError::None;
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

    const auto tensorCount     = GGUF::read_le_uint64(istream);
    const auto metadataKvCount = GGUF::read_le_uint64(istream);
    if( !tensorCount || !metadataKvCount ) { outError = ReadError::InvalidFormat; return tensorMap; }

    std::cout << "## Number of tensors:  " << *tensorCount     << std::endl;
    std::cout << "## Number of metadata: " << *metadataKvCount << std::endl;

    auto keyBuffer = std::make_unique<char[]>(MaximumKeyLength);
    for( int i=0 ; i<22 ; ++i ) {
        _add_metadata(tensorMap, istream, keyBuffer.get(), MaximumKeyLength);
    }

    return tensorMap;
}


//========================= WRITING TO GGUF FILE ==========================//





} // namespace tin