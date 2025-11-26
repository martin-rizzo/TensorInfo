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
 * Returns the relative name of this node based on a given prefix.
 *
 * This function takes a prefix and removes it from the node's name, returning
 * the remaining portion as a StringView.
 * 
 * If the prefix is not valid (i.e., the prefix does not match at the beginning 
 * of the node's name or does not cover a complete segment of the name), an 
 * empty string is returned.
 * 
 * @param prefix The prefix to be removed from the node's name.
 * @return A string view with the relative name after removing the prefix,
 *         or an empty string if the prefix is invalid.
 */
StringView
TensorTreeNode::relative_name(StringView prefix
) const noexcept {
    // if no prefix was provided, return the full name
    if( prefix.empty() ) { return name(); }

    // remove any trailing dot from the user provided prefix
    if( prefix.ends_with('.') ) { prefix.remove_suffix(1); }

    // if the prefix does not match at position 0,
    // it is invalid, return an empty string
    auto pos = name().find(prefix);
    if( pos != 0 ) { return ""; }

    // create a string_view from the node's name and remove the prefix
    StringView result{ name() };
    result.remove_prefix( prefix.size() );

    // if the resulting name does not start with a dot,
    // the prefix split a segment in half, return an empty string
    if( !result.starts_with('.') ) {
        return "";
    }
    // remove the leading dot and return the remaining relative name.
    result.remove_prefix(1);
    return result;
}

/**
 * Returns pointers to all subnodes of this node, ordered by name by default.
 *
 * If `recursive` is true, this method also includes subnodes from child
 * nodes in recursive manner.
 *
 * @param sortBy    The sorting criterion for the resulting list.
 * @param recursive Determines if only direct children of this node should be
 *                  included (`false`, default); or all subnodes should be
 *                  included (`true`).
 * @return
 *     A vector of pointers to `TensorTreeNode` objects representing the subnodes.
 */
const Vector<const TensorTreeNode*>
TensorTreeNode::subnode_pointers(SortBy sortBy,   // = SortBy::NAME
                                 bool   recursive // = false
) const noexcept {
    Vector<const TensorTreeNode*> result;
    SortBy currentOrder{ SortBy::NONE };

    // if not recursive, only extract immediate subnodes
    // (guaranteed to be in alphabetical-natural order)
    if( !recursive ) {
        result.reserve(_subnodeMap.size());
        for( const auto& [normalizedName, node] : _subnodeMap ) {
            result.push_back(&node);
        }
        currentOrder = SortBy::NAME;
    }
    // if recursive, collect all subnodes including descendants
    // (no order is guaranteed)
    else {
        result.reserve(1024); // arbitrary capacity estimation
        _collect_recursively(result);
        currentOrder = SortBy::NONE;
    }
    if( sortBy != SortBy::NONE && sortBy != currentOrder ) {
        /// TODO: implement node sorting by the "sortBy" parameter
    }
    return result;
}

/**
 * Returns pointers to all tensors of this node, ordered by name by default.
 *
 * If `recursive` is true, this method also includes tensors from child nodes
 * in recursive manner.
 *
 * @param sortBy    The sorting criterion for the resulting list.
 * @param recursive Determines if only tensors of this node should be
 *                  included (`false`, default); or tensors from all subnodes
 *                  should be included (`true`).
 * @return
 *     A vector of pointers to `TensorInfo` objects representing the tensors.
 */
const Vector<const TensorInfo*>
TensorTreeNode::tensor_pointers(SortBy sortBy,   // = SortBy::NAME
                                bool   recursive // = false
) const noexcept {
    Vector<const TensorInfo*> result;
    SortBy currentOrder{ SortBy::NONE };

    // if not recursive, only extract immediate tensors
    // (guaranteed to be in alphabetical-natural order)
    if( !recursive ) {
        result.reserve( _tensorMap.size() );
        for(const auto& [normalizedName, tensor] : _tensorMap) {
            result.push_back( &tensor );
        }
        currentOrder = SortBy::NAME;
    }
    // if recursive, collect all tensors including tensors from descendants
    // (no order is guaranteed)
    else {
        result.reserve(1024);
        _collect_recursively(result);
        currentOrder = SortBy::NONE;
    }
    if( sortBy != SortBy::NONE && sortBy != currentOrder ) {
        /// TODO: implement tensor sorting by the "sortBy" parameter
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
