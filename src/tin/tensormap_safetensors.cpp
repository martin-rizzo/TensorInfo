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
#include <iostream>   // for std::istream
#include <algorithm>  // for std::max
#include <tin/tensormap.h>
#include <rapidjson/document.h>
namespace tin {
namespace json = rapidjson;
using VectorSizeT = std::vector<size_t>;

// The maximum safe size for the header (in bytes).
// If the file is very large, this value can be increased.
static const size_t MaximumSafeHeaderSize = 10*1024*1024;

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
    if( !value.IsString() ) { return {}; }

    StringView valueStr{ value.GetString() };
    if( valueStr == "BOOL"    ) { return DType::BOOL;    }
    if( valueStr == "F4"      ) { return DType::F4;      }
    if( valueStr == "F6_E2M3" ) { return DType::F6_E2M3; }
    if( valueStr == "F6_E3M2" ) { return DType::F6_E3M2; }
    if( valueStr == "U8"      ) { return DType::U8;      }
    if( valueStr == "I8"      ) { return DType::I8;      }
    if( valueStr == "F8_E5M2" ) { return DType::F8_E5M2; }
    if( valueStr == "F8_E4M3" ) { return DType::F8_E4M3; }
    if( valueStr == "F8_E8M0" ) { return DType::F8_E8M0; }
    if( valueStr == "I16"     ) { return DType::I16;     }
    if( valueStr == "U16"     ) { return DType::U16;     }
    if( valueStr == "F16"     ) { return DType::F16;     }
    if( valueStr == "BF16"    ) { return DType::BF16;    }
    if( valueStr == "I32"     ) { return DType::I32;     }
    if( valueStr == "U32"     ) { return DType::U32;     }
    if( valueStr == "F32"     ) { return DType::F32;     }
    if( valueStr == "F64"     ) { return DType::F64;     }
    if( valueStr == "I64"     ) { return DType::I64;     }
    if( valueStr == "U64"     ) { return DType::U64;     }
    return DType::UNKNOWN;
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
 * Try to construct a TensorInfo object from JSON data.
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
_try_to_parse_tensor_info(StringView            name,
                          const json::Value&    value,
                          SharedPtr<const Path> ptrPath,
                          size_t                byteBufferOffset
){
    if( !value.IsObject() ) { return {}; }

    auto opt_dtype       = _read_dtype      ( value["dtype"]        );
    auto opt_shapeArray  = _read_sizet_array( value["shape"]        );
    auto opt_offsetArray = _read_sizet_array( value["data_offsets"] );
    if( !opt_dtype || !opt_shapeArray || !opt_offsetArray ) { return {}; }
    if( opt_offsetArray->size() != 2 ) { return {}; }
    return TensorInfo(name,
                      *opt_dtype,
                      Shape{ *opt_shapeArray },
                      ptrPath,
                      opt_offsetArray->at(0) + byteBufferOffset,
                      opt_offsetArray->at(1) + byteBufferOffset
                      );
}


//===================== READING FROM SAFETENSORS FILE =====================//

/**
 * Parses a safetensors file to construct a TensorMap.
 *
 * It constructs a map of tensors based on the information found within the
 * JSON-encoded header, including any metadata elements.
 *
 * @param firstBytes An array containing the first 8 bytes of a safetensors file.
 * @param istream    An input stream positioned right after the initial 8-byte header.
 * @param filePath   The full path of the original file used to create the input stream.
 *                   (If the `istream` does not originate from a file, this parameter should be empty)
 * @param fileSize   The total size of the file associated with the input stream.
 *                   (Can be 0 but no validation of offsets or sizes will be performed) 
 * @param byteBufferPosition The offset within the file where the raw tensor data buffer begins.
 *                           (If set to 0, this position will be auto-detected)
 *
 * @return A TensorMap object populated with theinformation from the safetensors file,
 *         or an empty map if any errors occur during processing.
 */
TensorMap
TensorMap::_fromsafetensors(const uint8_t   firstBytes[8],
                            std::istream&   istream,
                            const Path&     filePath,          // = {},
                            std::streamsize fileSize,          // = 0
                            std::streampos  byteBufferPosition // = 0
){
    TensorMap tensorMap;

    // first 8 bytes are header size (64-bit little endian)
    const size_t headerSize = (static_cast<size_t>(firstBytes[0]) <<  0) |
                              (static_cast<size_t>(firstBytes[1]) <<  8) |
                              (static_cast<size_t>(firstBytes[2]) << 16) |
                              (static_cast<size_t>(firstBytes[3]) << 24) |
                              (static_cast<size_t>(firstBytes[4]) << 32) |
                              (static_cast<size_t>(firstBytes[5]) << 40) |
                              (static_cast<size_t>(firstBytes[6]) << 48) |
                              (static_cast<size_t>(firstBytes[7]) << 56);

    // quickly calculates the maximum safe size of the header,
    // since a block of memory will be reserved for it later
    const size_t maxHeaderSize = fileSize>0
                        ? std::max<size_t>(MaximumSafeHeaderSize, fileSize/500)
                        : 10*MaximumSafeHeaderSize;
    if( headerSize > maxHeaderSize ) { return tensorMap;  }

    // allocate a block of memory for the header
    auto buffer = std::make_unique<char[]>(headerSize + 4);
    if( !buffer ) { return tensorMap; }

    // read header from istream
    istream.read(buffer.get(), headerSize);
    if( istream.fail() ) { return tensorMap; }

    // add 4 termination characters at the end of the buffer, just in case
    buffer[headerSize+0] = buffer[headerSize+1] = '\0';
    buffer[headerSize+2] = buffer[headerSize+3] = '\0';

    // auto-detect byte-buffer offset (if not provided)
    if( byteBufferPosition==0 ) {
        byteBufferPosition = !filePath.empty() ? istream.tellg() : std::streampos{0};
    }

    // in situ parsing the buffer into document
    // ATTENTION: buffer will be modified!
    json::Document document;
    document.ParseInsitu( buffer.get() );
    if( document.HasParseError() || !document.IsObject() ) {
        return tensorMap;
    }

    // the path that will be shared by all tensors
    auto ptrPath = std::make_shared<const Path>(filePath);

    // iterate over each element in the JSON document and insert a `TensorInfo`
    // for each one of them, except for "__metadata__" elements
    auto jsonRootDict = document.GetObject();
    for( const auto& jsonElement : jsonRootDict )
    {
        // the key/value pair must be a string/object pair
        if( !jsonElement.name.IsString()  ) { continue; }
        if( !jsonElement.value.IsObject() ) { continue; }

        StringView tensorName{ jsonElement.name.GetString() };
        if( tensorName == "__metadata__" )
        {
            /// TODO: parse checkpoint metadata
            continue;
        }
        // parse the TensorInfo from the JSON element
        // and insert it into the map
        auto tensorInfo = _try_to_parse_tensor_info(tensorName,
                                                    jsonElement.value,
                                                    ptrPath,
                                                    byteBufferPosition
                                                    );
        if ( tensorInfo ) {
            tensorMap.insert( *tensorInfo );
        }
    }
    return tensorMap;
}


//====================== WRITING TO SAFETENSORS FILE ======================//






} // namespace tin
