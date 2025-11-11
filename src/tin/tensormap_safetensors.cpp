/*
| File    : tensormap_gguf.cpp
| Purpose : TensorMap safetensors file reader/writer methods.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 9, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|
|  Safetensors Format Specification: https://github.com/huggingface/safetensors
|
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <iostream>
#include <tin/tensormap.h>
#include <rapidjson/document.h>
namespace tin {
namespace json = rapidjson;
using VectorSizeT = std::vector<size_t>;

//================================ HELPERS ================================//

/**
 * Reads a data type from a JSON value.
 *
 * This static function attempts to extract and interpret a data type name
 * from a given JSON `Value`. If the input is not valid or does not match
 * any known type, it returns an empty optional.
 *
 * @param value The JSON value containing the potential dtype as a string.
 * @return An optional containing the data type if successful, or an empty
 *         optional if unsuccessful.
 */
static Optional<DType>
_read_dtype(const json::Value& value
){
    if( value.IsString() ) {
        StringView dtypeStr{ value.GetString() };
        if( dtypeStr == "BOOL"    ) { return DType::BOOL;    }
        if( dtypeStr == "F4"      ) { return DType::F4;      }
        if( dtypeStr == "F6_E2M3" ) { return DType::F6_E2M3; }
        if( dtypeStr == "F6_E3M2" ) { return DType::F6_E3M2; }
        if( dtypeStr == "U8"      ) { return DType::U8;      }
        if( dtypeStr == "I8"      ) { return DType::I8;      }
        if( dtypeStr == "F8_E5M2" ) { return DType::F8_E5M2; }
        if( dtypeStr == "F8_E4M3" ) { return DType::F8_E4M3; }
        if( dtypeStr == "F8_E8M0" ) { return DType::F8_E8M0; }
        if( dtypeStr == "I16"     ) { return DType::I16;     }
        if( dtypeStr == "U16"     ) { return DType::U16;     }
        if( dtypeStr == "F16"     ) { return DType::F16;     }
        if( dtypeStr == "BF16"    ) { return DType::BF16;    }
        if( dtypeStr == "I32"     ) { return DType::I32;     }
        if( dtypeStr == "U32"     ) { return DType::U32;     }
        if( dtypeStr == "F32"     ) { return DType::F32;     }
        if( dtypeStr == "F64"     ) { return DType::F64;     }
        if( dtypeStr == "I64"     ) { return DType::I64;     }
        if( dtypeStr == "U64"     ) { return DType::U64;     }
    }
    return {};
}


/**
 * Reads an array of 64-bit unsigned integers from a JSON value.
 *
 * This static function attempt to extract and interpret an array of 64-bit
 * unsigned integers from a given JSON `Value`. If the input is not valid or
 * does not match the expected format, it returns an empty optional.
 *
 * @param value The JSON value containing the potential array of 64-bit unsigned integers.
 * @return An optional containing a vector with `size_t` values if successful,
 *         or an empty optional otherwise.
 */
static Optional<const VectorSizeT>
_read_sizet_array(const json::Value& value
){
    if( !value.IsArray() ) { return {}; }

    VectorSizeT array;
    for( const json::Value& integer : value.GetArray() ) {
        if( !integer.IsUint64() ) { return {}; }
        array.push_back( static_cast<size_t>(integer.GetUint64()) );
    }
    return array;
}


/**
 * Constructs TensorInfo from JSON data.
 *
 * This static function creates a `TensorInfo` object from a JSON representation.
 * It extracts the data type ('dtype'), shape array ('shape'), and data offsets
 * ('data_offsets') from the provided JSON `value`. If any component is missing
 * or malformed, it returns an empty optional.
 *
 * @param name              The name of the tensor (e.g., "encoder.block.0.layer.0.SelfAttention.k.weight")
 * @param value             The JSON value containing the tensor information in JSON format.
 * @param ptrPath           A shared pointer to the path where the tensor is stored.
 *                          (this can be `std::nullptr` for tensors that are not stored on disk)
 * @param byteBufferOffset  The offset of the "byte-buffer" within the safetensors file. 
 *                          (this can be 0 if the tensor is not stored on disk)
 * @return
 *    An optional containing the constructed tensor information if successful,
 *    or an empty optional if any component is missing or malformed.
 */
static Optional<TensorInfo>
_tensor_info_from_json(StringView            name,
                       const json::Value&    value,
                       SharedPtr<const Path> ptrPath,
                       size_t                byteBufferOffset)
{
    if( !value.IsObject() ) { return {}; }

    auto opDType       = _read_dtype      ( value["dtype"]        );
    auto opShapeArray  = _read_sizet_array( value["shape"]        );
    auto opOffsetArray = _read_sizet_array( value["data_offsets"] );
    if( !opDType || !opShapeArray || !opOffsetArray ) { return {}; }
    if( opOffsetArray->size() != 2 ) { return {}; }
    return TensorInfo(name,
                      *opDType,
                      Shape{ *opShapeArray },
                      ptrPath,
                      opOffsetArray->at(0) + byteBufferOffset,
                      opOffsetArray->at(1) + byteBufferOffset
                      );
}

//====================== READING A SAFETENSORS FILE =======================//

TensorMap
TensorMap::_fromsafetensors(const uint8_t firstBytes[8],
                            std::istream& istream,
                            const Path&   filePath,   // = {},
                            size_t        fileOffset  // = 0
){
    // first 8 bytes are header size (64-bit little endian)
    size_t headerSize = (static_cast<size_t>(firstBytes[0]) <<  0) |
                        (static_cast<size_t>(firstBytes[1]) <<  8) |
                        (static_cast<size_t>(firstBytes[2]) << 16) |
                        (static_cast<size_t>(firstBytes[3]) << 24) |
                        (static_cast<size_t>(firstBytes[4]) << 32) |
                        (static_cast<size_t>(firstBytes[5]) << 40) |
                        (static_cast<size_t>(firstBytes[6]) << 48) |
                        (static_cast<size_t>(firstBytes[7]) << 56);


    if( headerSize > 1024*1024 ) {
        throw std::runtime_error("Header size is too large.");
    }

    // the path that will be shared by all tensors
    auto ptrPath = std::make_shared<const Path>(filePath);

    // allocate memory for header
    auto buffer = std::make_unique<char[]>(headerSize + 4);
    if( !buffer ) { throw std::runtime_error("Failed to allocate memory for header.");  }

    // read header from istream
    istream.read(buffer.get(), headerSize); fileOffset+= headerSize;
    if( istream.fail() ) { throw std::runtime_error("Failed to read header."); }

    // add 4 termination characters at the end of the buffer, just in case
    buffer[headerSize+0] = buffer[headerSize+1] = '\0';
    buffer[headerSize+2] = buffer[headerSize+3] = '\0';

    // in situ parsing the buffer into document
    // ATTENTION: buffer will also be modified!
    json::Document document;
    document.ParseInsitu( buffer.get() );

    // iterate over each object in the JSON document and create a TensorInfo
    // for each one of them, except for the "__metadata__" object
    TensorMap tensorMap;
    for( auto& object : document.GetObject() ) {
        StringView name{ object.name.GetString() };
        if( name != "__metadata__" ) {
            auto tensorInfo = _tensor_info_from_json(name, object.value, ptrPath, fileOffset);
            if( tensorInfo.has_value() ) {
                std::cout << *tensorInfo << std::endl;
            }
        }
    }
    return tensorMap;
}

//================================ WRITING ================================//
 


} // namespace tin
