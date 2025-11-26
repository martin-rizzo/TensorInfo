/*
| File    : tensortreenode.cpp
| Purpose : 
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 24, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <tin/tensortreenode.h>
namespace tin {

//============================= QUERY METHODS =============================//

/**
 * Collects the subnodes of this node.
 *
 * This method returns a list of pointers to the subnodes of the current node.
 * The behavior depends on whether the operation is recursive or not:
 *  - If non-recursive, only the immediate subnodes are collected and returned
 *    in alphabetical order.
 *  - If recursive, all subnodes (including those from descendant nodes) are
 *    collected but without any guaranteed order.
 *
 * @param recursive  Determines if only direct children of this node should be
 *                   collected (false,default); or all subnodes should be
 *                   collected (true).
 * @return
 *     A Vector containing pointers to the collected TensorTreeNode objects.
 */
const Vector<const TensorTreeNode*>
TensorTreeNode::collect_subnodes(bool recursive // = false
) const noexcept {
    Vector<const TensorTreeNode*> result;

    // si no es recursivo, solo extraer los subnodos del nodo actual
    // se garantiza el orden alfabético-natural de los subnodos en la lista
    if( !recursive ) {
        result.reserve(_subnodeMap.size());
        for( const auto& [normalizedName, node] : _subnodeMap ) {
            result.push_back(&node);
        }
    }
    // si es recursivo, extraer todos los subnodos de este nodo y sus hijos
    // no se garantiza ningún orden en la lista de subnodos
    else {
        result.reserve(1024);
        _collect_recursively(result);
    }
    return result;
}

/**
 * Collects the tensor information objects within this node.
 *
 * This method returns a list of TensorInfo pointers that are contained within
 * the current node. The behavior depends on whether the operation is recursive
 * or not:
 *   - If non-recursive, only the tensors directly stored in this node are
 *     collected and returned in alphabetical order.
 *   - If recursive, all tensors (including those from descendant nodes) are
 *     collected but without any guaranteed order.
 *
 * @param recursive Determines if only direct tensors of this node should be
 *                  collected (false,default); or all tensors should be
 *                  collected (true).
 * @return
 *     A Vector containing pointers to the collected TensorInfo objects.
 */
const Vector<const TensorInfo*>
TensorTreeNode::collect_tensors(bool recursive // = false
) const noexcept {
    Vector<const TensorInfo*> result;

    // si no es recursivo, solo extraer los tensores del nodo actual
    // se garantiza el orden alfabético-natural de los tensores en la lista
    if( !recursive ) {
        result.reserve( _tensorMap.size() );
        for(const auto& [name, tensor] : _tensorMap) { result.push_back( &tensor ); }
    }
    // si es recursivo, extraer todos los tensores de este nodo y sus hijos
    // no se garantiza ningún orden en la lista de tensores
    else {
        result.reserve(1024);
        _collect_recursively(result);
    }
    return result;
}


//=========================== NODE MANIPULATION ===========================//

/**
 * Ensures the existence of a subnode with a given normalized name.
 *
 * This method returns a reference to the subnode with the specified normalized 
 * name (`normalizedName`). If such a subnode does not already exist, one is
 * created. The method then returns this (newly created or existing) subnode
 * for further manipulation or access. Note that the `name` parameter should be used when the actual
 * tensor's file name needs to differ from its normalized form.
 *
 * @param normalizedName The normalized name of the subnode to ensure, typically 
 *                       padded with zeros for natural sorting (e.g., "00004").
 *                       It should not contain any dot ('.') as this function
 *                       is non-recursive.
 * @param name           The original name of the subnode as it appears in files,
 *                       without normalization (e.g., "4").
 * @return A reference to the guaranteed TensorTreeNode subnode.
 */
TensorTreeNode&
TensorTreeNode::ensure_subnode(StringView normalizedNodeName, StringView nodeName) {
    auto it = _subnodeMap.find( normalizedNodeName );
    if( it == _subnodeMap.end() ) {
        auto subnodeName = name().empty() ? String{nodeName} : name() + '.' + String{nodeName};
        it = _subnodeMap.emplace(normalizedNodeName, TensorTreeNode{subnodeName}).first;
    }
    return it->second;
}

/**
 * Inserts a tensor into the tree based on its hierarchical normalized name.
 *
 * This method recursively constructs the node hierarchy according to segments
 * of `tensorNormalizedName`, where each segment represents a level in the
 * structure. Segments are separated by dots ('.'). If no dot is present in
 * the name, the tensor is directly inserted into this node.
 *
 * @param tensorNormalizedName The normalized hierarchical path within the tree,
 *                             using dots as separators. Numeric segments are
 *                             expected to be zero-padded for natural sorting.
 *                             (e.g., "model.layers.00004.input_layernorm.weight").
 * @param tensorName           The original name of the tensor as it appears in
 *                             file, without normalization.
 * @param tensor               Tensor information to insert into the tree.
 * @return `true` if the tensor is successfully inserted; `false` otherwise.
 */
bool
TensorTreeNode::insert_tensor(StringView        tensorNormalizedName,
                              StringView        tensorName,
                              const TensorInfo& tensor
){
    // find the position of the dot in both names to identify the first segment
    auto normalizedDotPos = tensorNormalizedName.find('.');
    auto dotPos           = tensorName.find('.');

    // If no dot is found in either name, add the tensor directly to this node
    if( normalizedDotPos == StringView::npos || dotPos == StringView::npos ) {
        return _tensorMap.emplace(tensorNormalizedName, tensor).second;
    }

    // extract the first segment of both names as the subnode names
    auto normalizedSubnodeName = tensorNormalizedName.substr(0, normalizedDotPos);
    auto subNodeName           = tensorName.substr(0, dotPos);    

    // ensure that the identified subnode exists
    auto& subnode = ensure_subnode(normalizedSubnodeName, subNodeName);

    // recursively insert the remaining part of the tensor name into the subnode
    return subnode.insert_tensor(
        tensorNormalizedName.substr(normalizedDotPos + 1), 
        tensorName.substr(dotPos + 1),
        tensor
    );
}

//============================= IMPLEMENATION =============================//

/**
 * Recursively collects subnodes and adds them to a given collector vector.
 *
 * This internal method is called by collect_subnodes() when recursion is
 * enabled. It iterates through the current node's subnodes, adding each
 * one to the provided collector Vector.
 *
 * @param collector A reference to a Vector where pointers to TensorTreeNode
 *                  objects will be accumulated.
 */
void
TensorTreeNode::_collect_recursively(Vector<const TensorTreeNode*>& collector
) const noexcept {
    for(const auto& [name, node] : _subnodeMap) {
        collector.push_back(&node);
        node._collect_recursively(collector);
    }
}

/**
 * Recursively collects tensors and adds them to a given collector vector.
 *
 * This internal method is called by collect_tensors() when recursion is
 * enabled. It iterates through the current node's subnodes, adding each
 * tensor to the provided collector Vector.
 *
 * @param collector A reference to a Vector where pointers to TensorInfo
 *                  objects will be accumulated.
 */
void
TensorTreeNode::_collect_recursively(Vector<const TensorInfo*>& collector
) const noexcept {
    for(const auto& [normalizedName, subnode] : _subnodeMap) {
        subnode._collect_recursively(collector);
    }
    for(const auto& [normalizedName, tensor] : _tensorMap) {
        collector.push_back( &tensor );
    }
}


} // namespace tin
