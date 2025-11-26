/*
| File    : tensortree.cpp
| Purpose : Organize tensors in a hierarchical tree structure.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <tin/tensortree.h>
namespace tin {

//======================= CONSTRUCTION/DESTRUCTION ========================//

/**
 * Constructs a TensorTree from a given TensorMap.
 * @param map A reference to the TensorMap containing tensor information.
 */
TensorTree::TensorTree(const TensorMap& map
): _root{ "" }
{
    for(auto& [name, info] : map) {
        insert(info);
    }
}


} // namespace tin