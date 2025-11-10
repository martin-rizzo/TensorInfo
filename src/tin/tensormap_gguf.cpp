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
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <tin/tensormap.h>
namespace tin {

//================================ READING ================================//

TensorMap
TensorMap::_fromgguf(std::istream& istream,
                     const Path& path, // = {},
                     size_t offset     // = 0
){
    TensorMap tensorMap;


    return tensorMap;
}

//================================ WRITING ================================//
 



} // namespace tin