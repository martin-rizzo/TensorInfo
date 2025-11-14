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
 * Attempts to insert tensor information into the provided `tensorMap`.
 *
 * This function extracts necessary details from a JSON object representing
 * information about a tensor (e.g., its data type, shape, etc). If all
 * required components are valid, it constructs a `TensorInfo` object
 * and inserts it into the `tensorMap` using the provided name.
 *
 * @param[in] tensorMap  The map where tensor information will be stored.
 * @param[in] jsonName   JSon object containing the tensor's name.
 * @param[in] jsonValue  JSON object containing the tensor's data type, shape, etc
 * @param[in] ptrPath    Shared pointer to a Path object, representing the path to
 *                       file where the tensor raw data is stored. Can be `nullptr`
 * @param[in] byteBufferOffset
 *                       Offset within the file where the raw-data is stored.
 *                       This parameter is NOT the offset of the tensor data in the
 *                       file, but the offset where the main data block starts.
 */
static void
_insert_tensor_info(TensorMap&            tensorMap,
                    const json::Value&    jsonName,
                    const json::Value&    jsonValue,
                    SharedPtr<const Path> ptrPath,
                    size_t                byteBufferOffset
){
    if( !jsonName.IsString()  ) { return; }
    if( !jsonValue.IsObject() ) { return; }

    auto name        = StringView{ jsonName.GetString() };
    auto dtype       = _read_dtype      ( jsonValue["dtype"]        );
    auto shapeArray  = _read_sizet_array( jsonValue["shape"]        );
    auto offsetArray = _read_sizet_array( jsonValue["data_offsets"] );
    if( name.empty() || !dtype || !shapeArray || !offsetArray ) { return; }
    if( offsetArray->size() != 2 ) { return; }

    tensorMap.insert( TensorInfo{ name,
                                  *dtype,
                                  Shape{ *shapeArray },
                                  ptrPath,
                                  static_cast<std::streampos>(offsetArray->at(0) + byteBufferOffset),
                                  static_cast<std::streampos>(offsetArray->at(1) + byteBufferOffset)
    });
}

/**
 * Attempts to set a new metadata key/value pair in the `tensorMap`.
 *
 * This function updates or inserts metadata into the `tensorMap` if both
 * provided JSON values represent valid strings.
 *
 * @param[in] tensorMap  The map where the metadata will be inserted.
 * @param[in] jsonName   JSON object containing the metadata name.
 * @param[in] jsonValue  JSON object containing the metadata value.
 */
static void
_insert_checkpoint_metadata(TensorMap&         tensorMap,
                            const json::Value& jsonName,
                            const json::Value& jsonValue
){
    if( !jsonName.IsString()  ) { return; }
    if( !jsonValue.IsString() ) { return; }

    auto name  = StringView{ jsonName.GetString() };
    auto value = StringView{ jsonValue.GetString() };
    if( name.empty() ) { return; }

    tensorMap.metadata().set_string( name, value );
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
) noexcept {
    TensorMap tensorMap;

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
    if( headerSize > maxHeaderSize ) { outError = ReadError::HeaderTooLarge; return tensorMap;  }

    // allocate a block of memory for the header
    auto buffer = std::make_unique<char[]>(headerSize + 4);
    if( !buffer ) { outError = ReadError::MemoryAllocationFailed; return tensorMap; }

    // read header from istream
    istream.read(buffer.get(), headerSize);
    if( istream.fail() ) { outError = ReadError::InvalidFormat; return tensorMap; }

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
        outError = ReadError::InvalidFormat; return tensorMap;
    }

    // the path that will be shared by all tensors
    auto ptrPath = std::make_shared<const Path>(filePath);

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
            // be an object containing the metadata elements in key/value pairs.
            auto jsonMetadataDict = jsonElement.value.GetObject();
            for( const auto& jsonMetadataKV : jsonMetadataDict )
            {
                _insert_checkpoint_metadata(tensorMap,
                                            jsonMetadataKV.name,
                                            jsonMetadataKV.value
                                            );
            }
        }
        else
        {
            // insert a new TensorInfo entry into the map
            _insert_tensor_info(tensorMap,
                                jsonElement.name,
                                jsonElement.value,
                                ptrPath, byteBufferPosition
                                );
       }
    }
    return tensorMap;
}


//====================== WRITING TO SAFETENSORS FILE ======================//






} // namespace tin
