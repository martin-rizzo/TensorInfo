/*
| File    : tensormap.h
| Purpose : Maps tensor names to their metadata (TensorInfo)
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <fstream>     // for std::ifstream
#include <tin/tensormap.h>
namespace tin {

//=========================== LOADING TENSORMAP ===========================//

/**
 * Creates a TensorMap from a file.
 *
 * This function attempts to read and create a `TensorMap` object from the
 * file located at `filePath`. If an error occurs during this process, the
 * `outError` parameter is set accordingly, and an empty `TensorMap` is
 * returned.
 *
 * @param      filePath The path to the .safetensors or .gguf file.
 * @param[out] outError The variable that receives the error code.
 *                      (It will be set to `ReadError::None` if no error occurs)
 * @return
 *     A populated TensorMap object if the file is successfully read and parsed,
 *     or an empty TensorMap in case of errors.
 */
TensorMap
TensorMap::from_file(const Path& filePath,
                     ReadError&  outError
) noexcept {
    if( !std::filesystem::exists(filePath) ) {
        outError = ReadError::FileNotFound; return TensorMap{};
    }
    auto istream  = std::ifstream{ filePath, std::ios_base::binary };
    auto fileSize = static_cast<std::streampos>( std::filesystem::file_size(filePath) );
    return TensorMap::from_stream(istream, outError, filePath, fileSize);
}

/**
 * Creates a TensorMap from an input stream.
 *
 * This function attempts to create a `TensorMap` object by reading data from
 * the provided input stream `istream`. It determines whether the file is in
 * GGUF or safetensors format. If an error occurs during this process, the
 * `outError` parameter is set accordingly.
 *
 * @param      istream  The input stream from which data will be read.
 * @param[out] outError The variable that receives the error code.
 *                      (It will be set to `ReadError::None` if no error occurs)
 * @param      filePath The full path of the original file used to create the input stream.
 *                      (If the `istream` does not originate from a file, this parameter should be empty)
 * @param      fileSize The total size of the file associated with the input stream.
 *                      (Can be 0 but no validation of offsets or sizes will be performed) 
 * @return
 *     A populated TensorMap object if the file is successfully read and parsed,
 *     or an empty TensorMap in case of errors.
 */
TensorMap
TensorMap::from_stream(std::istream&   istream,
                       ReadError&      outError,
                       const Path&     filePath,  // = {},
                       std::streamsize fileSize   // = 0
) noexcept {

    char     firstChars[8];
    uint8_t* firstBytes = reinterpret_cast<uint8_t*>(firstChars);

    // read the first 8 bytes of istream
    istream.read(firstChars, 8);
    const auto ggufIdentifier = std::string_view(firstChars, 4);
    const auto ggufVersion    = firstBytes[4] + firstBytes[5] * 256 + firstBytes[6] * (256*256) + firstBytes[7] * (256*256*256);


    outError = ReadError::None;
    if( ggufIdentifier == "GGUF" && 1 <= ggufVersion && ggufVersion <= 5)  {
        return _fromgguf(firstBytes, istream, outError, filePath, fileSize);
    } else {
        return _fromsafetensors(firstBytes, istream, outError, filePath, fileSize);
    }
}




} // namespace tin