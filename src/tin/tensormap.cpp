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
    auto istream = std::ifstream(path);
    return TensorMap::from_stream(istream, path);
}

TensorMap
TensorMap::from_stream(std::istream& istream,
                       const Path&   path,   // = {},
                       size_t        offset  // = 0)
){
    /// TODO: determine if the data format of istream is safetensors or gguf
    const bool is_safetensors = true;
    return is_safetensors
        ? _fromsafetensors(istream, path, offset)
        : _fromgguf(istream, path, offset);
}




} // namespace tin