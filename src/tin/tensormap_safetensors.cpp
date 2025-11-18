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
 * Appends a checkpoint metadata entry in the provided metadata container.
 *
 * This function modifies the `metadata` container by adding a new key-value pair.
 * If the specified key already exists, its corresponding value is updated.
 * If `jsonName` or `jsonValue` are not strings, no action is taken and the
 * function returns without making any changes.
 *
 * @param metadata   The Metadata container that will be updated.
 * @param jsonName   A JSON object containing the name (key) of the metadata.
 * @param jsonValue  A JSON object containing the value associated with the metadata key. 
 */
static void
_append_checkpoint_metadata(Metadata&          metadata,
                            const json::Value& jsonName,
                            const json::Value& jsonValue
){
    if( !jsonName.IsString()  ) { return; }
    if( !jsonValue.IsString() ) { return; }
    auto name  = StringView{ jsonName.GetString() };
    auto value = StringView{ jsonValue.GetString() };
    if( name.empty() ) { return; }
    metadata.set_string( name, value );
}

/**
 * Adds a TensorInfo entry to the provided vector.
 *
 * This function processes the given JSON values to extract necessary tensor
 * attributes, including name, data type, shape, and data offsets. If these
 * attributes are valid, it constructs a new `TensorInfo` object and appends
 * it to the specified `tensors` vector.
 * The function returns without making any changes if any of the checks fail.
 *
 * @param tensors   A reference to the vector of TensorInfo objects which will be updated.
 * @param jsonName  A JSON value expected to contain the name (key) for the tensor.
 * @param jsonValue A JSON object containing attributes like dtype, shape, and data_offsets associated with the tensor.
 * @param ptrPath   A shared pointer to a Path object indicating the source file path.
 */
static void
_append_tensor_info(std::vector<TensorInfo>& tensors,
                    const json::Value&       jsonName,
                    const json::Value&       jsonValue,
                    SharedPtr<const Path>    ptrPath
){
    if( !jsonName.IsString()  ) { return; }
    if( !jsonValue.IsObject() ) { return; }

    auto name        = StringView{ jsonName.GetString() };
    auto dtype       = _read_dtype      ( jsonValue["dtype"]        );
    auto shapeArray  = _read_sizet_array( jsonValue["shape"]        );
    auto offsetArray = _read_sizet_array( jsonValue["data_offsets"] );
    if( name.empty() || !dtype || !shapeArray || !offsetArray ) { return; }
    if( offsetArray->size() != 2 ) { return; }

    tensors.emplace_back( name,
                          *dtype,
                          Shape{ *shapeArray },
                          ptrPath,
                          static_cast<std::streampos>(offsetArray->at(0)),
                          static_cast<std::streampos>(offsetArray->at(1))
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
                            ReadError&      outError,
                            const Path&     filePath,          // = {},
                            std::streamsize fileSize,          // = 0
                            std::streampos  byteBufferPosition // = 0
) noexcept
{
    // first 8 bytes are header size (64-bit little endian)
    const size_t headerSize{ (static_cast<size_t>(firstBytes[0]) <<  0) |
                             (static_cast<size_t>(firstBytes[1]) <<  8) |
                             (static_cast<size_t>(firstBytes[2]) << 16) |
                             (static_cast<size_t>(firstBytes[3]) << 24) |
                             (static_cast<size_t>(firstBytes[4]) << 32) |
                             (static_cast<size_t>(firstBytes[5]) << 40) |
                             (static_cast<size_t>(firstBytes[6]) << 48) |
                             (static_cast<size_t>(firstBytes[7]) << 56)
                           };

    // quickly calculates the maximum safe size of the header,
    // since a block of memory will be reserved for it later
    const size_t maxHeaderSize = fileSize>0
                        ? std::max<size_t>(MaximumSafeHeaderSize, fileSize/500)
                        : 10*MaximumSafeHeaderSize;
    if( headerSize > maxHeaderSize ) { outError = ReadError::HeaderTooLarge; return {};  }

    // allocate a block of memory for the header
    auto buffer = std::make_unique<char[]>(headerSize + 4);
    if( !buffer ) { outError = ReadError::MemoryAllocationFailed; return {}; }

    // read header from istream
    istream.read(buffer.get(), headerSize);
    if( istream.fail() ) { outError = ReadError::InvalidFormat; return {}; }

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
        outError = ReadError::InvalidFormat; return {};
    }

    // the path that will be shared by all tensors
    auto ptrPath = std::make_shared<const Path>(filePath);

    Metadata                metadata;
    std::vector<TensorInfo> tensors;

    // iterate over each element within the JSON document, creating a `TensorInfo`
    // entry for each one. Element named "__metadata__" is handled separately.
    auto jsonRootDict = document.GetObject();
    for( const auto& jsonElement : jsonRootDict )
    {
        // each key/value pair must be a string/object pair
        if( !jsonElement.name.IsString()  ) { continue; }
        if( !jsonElement.value.IsObject() ) { continue; }

        StringView tensorName{ jsonElement.name.GetString() };
        if( tensorName == "__metadata__"  )
        {
            // handling for "__metadata__" element, this element is expected to
            // be an object containing the metadata elements in key/value pairs
            auto jsonMetadataDict = jsonElement.value.GetObject();
            for( const auto& jsonMetadataKV : jsonMetadataDict ) {
                // append the key/value pair as checkpoint metadata
                _append_checkpoint_metadata( metadata,
                                             jsonMetadataKV.name,
                                             jsonMetadataKV.value
                                             );
            }
        }
        else
        {
            // handling for tensor element, this element is expected to contain
            // the name of a tensor, and an object value containing tensor info
            _append_tensor_info( tensors,
                                 jsonElement.name,
                                 jsonElement.value,
                                 ptrPath
                                 );
       }
    }
    // create the final `TensorMap` object from the parsed tensors and metadata
    return { std::move(tensors), std::move(metadata), byteBufferPosition };
}


//====================== WRITING TO SAFETENSORS FILE ======================//

/// TODO: implement writing to safetensors file


} // namespace tin
