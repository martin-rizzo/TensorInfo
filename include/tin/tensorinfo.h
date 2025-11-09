/*
| File    : <tin/tensorinfo.h>
| Purpose : Store all metadata about a tensor: name, shape, data type,
|           even where the raw data of a tensor is located, etc.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_TENSORINFO_H_
#define TIN_TENSORINFO_H_
#include <tin/common.h> // for String, StringView, FilePath, SharedPtr
#include <tin/dtype.h>
#include <tin/shape.h>
#include <tin/unnamedtensorinfo.h>
namespace tin {


class TensorInfo
{
public:
    TensorInfo(StringView name, DType dtype, const Shape& shape, StringView filePath, size_t begin, size_t end);
    TensorInfo(StringView name, DType dtype, const Shape& shape, SharedPtr<const Path> ptrPath, size_t begin, size_t end);
    TensorInfo(StringView name, const TensorInfo& other);
private:
    TensorInfo(SharedPtr<String> ptrName, SharedPtr<UnnamedTensorInfo> ptrUnnamedTensorInfo);

public:
    [[nodiscard]] DType        dtype() const { return _ptrUnnamedTensorInfo->dtype(); }
    [[nodiscard]] const Shape& shape() const { return _ptrUnnamedTensorInfo->shape(); }
    [[nodiscard]] size_t       begin() const { return _ptrUnnamedTensorInfo->begin(); }
    [[nodiscard]] size_t       end()   const { return _ptrUnnamedTensorInfo->end();   }
    [[nodiscard]] size_t       size()  const { return _ptrUnnamedTensorInfo->size();  }

public:
    [[nodiscard]] String to_string() const;

// IMPLEMENTATION
private:
    SharedPtr<String>            _ptrName;               ///< A pointer to the name of this tensor
    SharedPtr<UnnamedTensorInfo> _ptrUnnamedTensorInfo;  ///< A pointer to an unnamed tensor info
};


} // namespace tin


/**
 * Overloads the insertion (<<) operator for outputting tin::DType values to streams.
 *
 * This inline function allows objects of type `tin::DType` to be directly
 * inserted into output streams such as `std::cout`.
 *
 * @param os   The output stream where the thin::DType will be written.
 * @param type The tin::DType value to be inserted into the output stream.
 * @return A reference to 'os' for chaining.
 */
inline 
std::ostream& operator<<(std::ostream& os, const tin::TensorInfo& tensorInfo) {
    return os << tensorInfo.to_string();
}



#endif // TIN_TENSORINFO_H_
