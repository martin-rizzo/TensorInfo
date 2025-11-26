/*
| File    : tensortree.h
| Purpose : Organize tensors in a hierarchical tree structure.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_TENSORTREE_H_
#define TIN_TENSORTREE_H_
#include <tin/tensormap.h>
#include <tin/tensortreenode.h>
namespace tin {


/**
 * The checkpoint tensor information represented as a hierarchical tree structure.
 */
class TensorTree
{
// CONSTRUCTION/DESTRUCTION
public:
    TensorTree(const TensorMap& map);
    TensorTree() = default;
    ~TensorTree() = default;

// ACCESORS
public:
    const TensorTreeNode& root() const noexcept;

// MODIFIERS
public:
    bool insert(const TensorInfo& tensorInfo);
    int  flatten_single_tensor_subnodes();


// IMPLEMENTATION
private:
    TensorTreeNode _root;
};


//================================ INLINES ================================//

/**
 * Returns a reference to the root node of the tensor tree.
 *
 * This function provides read-only access to the root node of the tensor tree
 * structure. The returned reference is constant, ensuring that the underlying
 * data cannot be modified through this accessor.
 * 
 * @return The root node of the tensor tree.
 */
inline const TensorTreeNode&
TensorTree::root() const noexcept { return _root; }

/**
 * Inserts a new tensor into the tensor tree structure.
 *
 * This function adds a new tensor represented by the given TensorInfo object
 * into the appropriate position within the tensor tree. The insertion is
 * performed based on the normalized name of the tensor, ensuring natural
 * ordering within the each node.
 *
 * @param tensorInfo The TensorInfo object to be inserted into the tree.
 * @return `true` if the insertion is successful, `false` otherwise
 *         (e.g., duplicate tensor names).
 */ 
inline bool
TensorTree::insert(const TensorInfo& tensorInfo) {
    return _root.insert_tensor(tensorInfo.generate_normalized_name(),
                               tensorInfo.name(),
                               tensorInfo
                               );
}

/**
 * Flattens all single tensor subnodes within the entire tree.
 *
 * This method traverses the entire TensorTree, identifying and removing any 
 * subnode that contains only one tensor. The contained tensor is then moved up
 * to its parent node, effectively flattening the structure wherever possible.
 * This process helps in simplifying the overall tree by eliminating unnecessary
 * hierarchical levels where a subnode does not add meaningful groupings beyond
 * a single tensor.
 * @return
 *   The total number of single-tensor subnodes that were flattened.
 */
inline int
TensorTree::flatten_single_tensor_subnodes() {
    return _root.flatten_single_tensor_subnodes(true);
}


}      // namespace tin
#endif // TIN_TENSORTREE_H_
