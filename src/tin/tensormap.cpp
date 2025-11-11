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

//======================= CONSTRUCTION/DESTRUCTION ========================//


//================================ LOADING ================================//


TensorMap
TensorMap::from_file(const Path& path)
{
    if( !std::filesystem::exists(path) ) {
        throw std::runtime_error("File not found");
    }
    auto istream = std::ifstream(path, std::ios_base::binary);
    return TensorMap::from_stream(istream, path);
}

TensorMap
TensorMap::from_stream(std::istream& istream,
                       const Path&   path,   // = {},
                       size_t        offset  // = 0)
){

    char     firstChars[8];
    uint8_t* firstBytes = reinterpret_cast<uint8_t*>(firstChars);

    // read the first 8 bytes of istream
    istream.read(firstChars, 8);
    const auto ggufIdentifier = std::string_view(firstChars, 4);
    const auto ggufVersion    = firstBytes[4] + firstBytes[5] * 256 + firstBytes[6] * (256*256) + firstBytes[7] * (256*256*256);


    if( ggufIdentifier == "GGUF" && 1 <= ggufVersion && ggufVersion <= 5)  {
        return _fromgguf(firstBytes, istream, path, offset+8);
    }
    return _fromsafetensors(firstBytes, istream, path, offset+8);
}




} // namespace tin