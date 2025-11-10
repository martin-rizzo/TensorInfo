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
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <tin/tensormap.h>
namespace tin {

//================================ READING ================================//

TensorMap
TensorMap::_fromsafetensors(std::istream& istream,
                            const Path& path,   // = {},
                            size_t offset       // = 0
){
    TensorMap tensorMap;

   auto tensorInfo = TensorInfo{ "encoder.block.0.layer.0.SelfAttention.k.weight", DType::F16, {4096, 4096}, "dir/file.safetensors", 0, 100 };
   tensorMap.insert(tensorInfo);

    return tensorMap;
}

//================================ WRITING ================================//
 


} // namespace tin
