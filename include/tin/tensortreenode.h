/*
| File    : tensortreenode.h
| Purpose : 
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 24, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_TENSORTREENODE_H_
#define TIN_TENSORTREENODE_H_
#include <map>        // for std::map
#include <functional> // for std::less
#include <tin/common.h>
#include <tin/sortby.h>
#include <tin/tensorinfo.h>
namespace tin {


// node in an tree structure where leaves are TensorInfo objects
class TensorTreeNode
{
public:
    using StringToNodeMap       = std::map<String, TensorTreeNode, std::less<>>;
    using StringToTensorInfoMap = std::map<String, TensorInfo, std::less<>>;

// CONSTRUCTION/DESTRUCTION
public:
    TensorTreeNode(StringView name);
    ~TensorTreeNode() = default;

// QUERY METHODS
public:
    const String&                       name() const noexcept;
    const Vector<const TensorTreeNode*> collect_subnodes(bool recursive = false) const noexcept;
    const Vector<const TensorInfo*>     collect_tensors(bool recursive = false) const noexcept;

// NODE MANIPULATION
public:
    TensorTreeNode& ensure_subnode(StringView normalizedName, StringView name);
    bool            insert_tensor(StringView normalizedPath, StringView tensorPath, const TensorInfo& tensor);
    
// IMPLEMENTATION
private:
    void _collect_recursively(Vector<const TensorTreeNode*>& collector) const noexcept;
    void _collect_recursively(Vector<const TensorInfo*>& collector) const noexcept;
private:
    String                _name;
    StringToNodeMap       _subnodeMap;
    StringToTensorInfoMap _tensorMap;
};


//======================== INLINES: QUERY METHODS =========================//

inline
TensorTreeNode::TensorTreeNode(StringView name
): _name{ name }
{ }

inline const String&
TensorTreeNode::name() const noexcept {
    return _name;
}


}      // namespace tin
#endif // TIN_TENSORTREENODE_H_
