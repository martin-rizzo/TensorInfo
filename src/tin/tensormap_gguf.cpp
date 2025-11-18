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


/**
 * Converts a GGMLType to its corresponding DType.
 *
 * This function takes a value of type `GGMLType` and maps it to the equivalent
 * `DType`. If no match is found, the function returns `DType::UNKNOWN`.
 *
 * @param ggmltype The GGMLType enumeration value to be converted.
 * @return
 *   The corresponding `DType` for the provided GGMLType,
 *   or DType::UNKNOWN if the input does not match any supported type.
 */
static DType
dtype_from_ggmltype(GGMLType ggmltype) noexcept {
    switch( ggmltype ) {
        case GGMLType::F32    : return DType::F32;
        case GGMLType::F16    : return DType::F16;
        case GGMLType::Q4_0   : return DType::Q4_0;
        case GGMLType::Q4_1   : return DType::Q4_1;
        case GGMLType::Q5_0   : return DType::Q5_0;
        case GGMLType::Q5_1   : return DType::Q5_1;
        case GGMLType::Q8_0   : return DType::Q8_0;
        case GGMLType::Q8_1   : return DType::Q8_1;
        case GGMLType::Q2_K   : return DType::Q2_K;
        case GGMLType::Q3_K   : return DType::Q3_K;
        case GGMLType::Q4_K   : return DType::Q4_K;
        case GGMLType::Q5_K   : return DType::Q5_K;
        case GGMLType::Q6_K   : return DType::Q6_K;
    ////case GGMLType::Q8_K   : return DType::Q8_K;
        case GGMLType::IQ2_XXS: return DType::IQ2_XXS;
        case GGMLType::IQ2_XS : return DType::IQ2_XS;
        case GGMLType::IQ3_XXS: return DType::IQ3_XXS;
        case GGMLType::IQ1_S  : return DType::IQ1_S;
        case GGMLType::IQ4_NL : return DType::IQ4_NL;
        case GGMLType::IQ3_S  : return DType::IQ3_S;
        case GGMLType::IQ2_S  : return DType::IQ2_S;
        case GGMLType::IQ4_XS : return DType::IQ4_XS;
        case GGMLType::I8     : return DType::I8;
        case GGMLType::I16    : return DType::I16;
        case GGMLType::I32    : return DType::I32;
        case GGMLType::I64    : return DType::I64;
        case GGMLType::F64    : return DType::F64;
        case GGMLType::IQ1_M  : return DType::IQ1_M;
        case GGMLType::BF16   : return DType::BF16;
    ////case GGMLType::TQ1_0  : return DType::TQ1_0;
    ////case GGMLType::TQ2_0  : return DType::TQ2_0;
    ////case GGMLType::MXFP4  : return DType::MXFP4;
        default: break;
    }
    return DType::UNKNOWN;
}

/**
 * Appends a key-value pair into a Metadata collection.
 *
 * This function reads a key and its corresponding value from an `GGUFIStream`,
 * and then inserts this data as a key-value pair into the provided `Metadata`
 * collection.
 *
 * @param metadata The Metadata collection where the key-value will be stored.
 * @param istream  An GGUF input stream from which key and value are read.
 * @return `ReadError::None` on success, or an error code indicating the type of failure.
 */
static ReadError
_append_checkpoint_metadata(Metadata&    metadata,
                            GGUFIStream& istream
) noexcept {
    // read the key
    auto key = istream.read_string();
    if( !key ) { return ReadError::MissingData; }

    // read the value type
    // (each entry in the metadata can be of a different type)
    auto metadataValueType = istream.read_metadata_value_type();
    if( !metadataValueType ) { return ReadError::MissingData; }
    switch( *metadataValueType )
    {
        // 1-byte value where 0 is false and 1 is true.
        // anything else is invalid, and should be treated as either
        // the model being invalid or the reader being buggy.
        case GGUFMetadataValueType::BOOL: {
            auto value = istream.read_uint8();
            if( !value )      { return ReadError::MissingData;   }
            if( *value >= 2 ) { return ReadError::InvalidFormat; }
            metadata.set_boolean(*key, *value!=0, StorageType::UINT8);
            } break;

        case GGUFMetadataValueType::INT8: {
            auto value = istream.read_int8();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_integer(*key, *value, StorageType::INT8);
            } break;

        case GGUFMetadataValueType::UINT8: {
            auto value = istream.read_uint8();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_integer(*key, *value, StorageType::UINT8);
            } break;

        case GGUFMetadataValueType::INT16: {
            auto value = istream.read_le_int16();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_integer(*key, *value, StorageType::INT16);
            } break;

        case GGUFMetadataValueType::UINT16: {
            auto value = istream.read_le_uint16();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_integer(*key, *value, StorageType::UINT16);
            } break;

        case GGUFMetadataValueType::INT32: {
            auto value = istream.read_le_int32();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_integer(*key, *value, StorageType::INT32);
            } break;

        case GGUFMetadataValueType::UINT32: {
            auto value = istream.read_le_uint32();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_integer(*key, *value, StorageType::UINT32);
            } break;

        case GGUFMetadataValueType::INT64: {
            auto value = istream.read_le_int64();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_integer(*key, *value, StorageType::INT64);
            } break;

        case GGUFMetadataValueType::UINT64: {
            auto value = istream.read_le_uint64();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_integer(*key, *value, StorageType::UINT64);
            } break;

        case GGUFMetadataValueType::FLOAT32: {
            auto value = istream.read_float32();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_float(*key, *value, StorageType::FLOAT32);
            } break;

        case GGUFMetadataValueType::FLOAT64: {
            auto value = istream.read_float64();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_float(*key, *value, StorageType::FLOAT64);
            } break;

        case GGUFMetadataValueType::STRING: {
            auto value = istream.read_string();
            if( !value ) { return ReadError::MissingData; }
            metadata.set_string(*key, *value);
            } break;

        case GGUFMetadataValueType::ARRAY:
            // DEBUG
            std::cerr << "## ignoring: " << *key << std::endl;
            if( !istream.ignore_array() ) { return ReadError::MissingData; }
            break;
    }
    return ReadError::None;
}

/**
 * Appends tensor information read from an input stream to a vector.
 *
 * This function reads information about a tensor from an GGUF input stream,
 * converts it into a `TensorInfo` object, and appends this object to the
 * provided vector.
 *
 * @param tensors  The vector of TensorInfo objects where the new one will be appended.
 * @param istream  An GGUF input stream providing serialized information.
 * @param ptrPath  A shared pointer to a Path object indicating the source file path.
 * @return `ReadError::None` on success, or an error code indicating the type of failure.
 */
static ReadError
_append_tensor_info(std::vector<TensorInfo>& tensors,
                    GGUFIStream&             istream,
                    SharedPtr<const Path>    ptrPath
) noexcept {
    // read tensor name
    auto name = istream.read_string();
    if( !name ) { return ReadError::MissingData; }

    // read "shape" as an array of ULongs (uint64_t)
    auto shapeArraySize = istream.read_le_uint32();
    if( !shapeArraySize      ) { return ReadError::MissingData;   }
    if( *shapeArraySize <= 0 ) { return ReadError::InvalidFormat; }
    if( *shapeArraySize >  8 ) { return ReadError::InvalidFormat; }
    const auto shapeArray = istream.read_seq_of_le_uint64( *shapeArraySize );
    if( !shapeArray ) { return ReadError::MissingData; }

    // read ggml-type and convert to `DType`
    auto ggmlType = istream.read_ggml_type();
    if( !ggmlType ) { return ReadError::MissingData; }
    auto dtype = dtype_from_ggmltype(*ggmlType);

    // read the `rawDataBegin` offset
    auto offset = istream.read_le_uint64();
    if( !offset ) { return ReadError::MissingData; }

    // create the tensor info and append it to the vector
    tensors.emplace_back( *name,
                          dtype,
                          Shape{ *shapeArray },
                          ptrPath,
                          static_cast<std::streampos>(*offset), // rawDataBegin
                          0                                     // rawDataEnd (will be set later)
                        );
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
        outError = ReadError::InvalidFormat; return {};
    }
    if( ggufVersion != 3 ) {
        outError = ReadError::UnsupportedVersion; return {};
    }


    Metadata                metadata;
    std::vector<TensorInfo> tensors;

    // the path that will be shared by all tensors
    auto ptrPath = std::make_shared<const Path>(filePath);

    // read number of tensors and number of key-value pairs in metadata
    const auto tensorCount     = istream.read_le_uint64();
    const auto metadataKvCount = istream.read_le_uint64();
    if( !tensorCount || !metadataKvCount ) {
        outError = ReadError::InvalidFormat; return {};
    }
    // read all key-value pairs in the "metadata" section
    for( ULong i=0 ; i<metadataKvCount && outError==ReadError::None ; ++i ) {
        outError = _append_checkpoint_metadata(metadata, istream);
    }
    // read all tensor infos in the "tensors" section
    for( ULong i=0 ; i<tensorCount && outError==ReadError::None ; ++i ) {
        outError = _append_tensor_info(tensors, istream, ptrPath);
    }
    // return the final `TensorMap` object from the parsed tensors and metadata
    return { std::move(tensors), std::move(metadata), byteBufferPosition };
}


//========================= WRITING TO GGUF FILE ==========================//





} // namespace tin