/*
| File    : <tin/unnamedtensorinfo.h>
| Purpose : Stores the metadata of a tensor (shape, dtype, etc.) but not its name!
|           This class is generally not used directly, it is intended for internal use by TensorInfo.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 8, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_UNNAMEDTENSORINFO_H_
#define TIN_UNNAMEDTENSORINFO_H_
#include <tin/common.h> // for String, StringView, Path, SharedPtr
#include <tin/dtype.h>
#include <tin/shape.h>
namespace tin {


/**
 * Encapsulates the metadata associated with a tensor.
 * 
 * This class is primarily intended for internal use within other components,
 * like `TensorInfo`, rather than being utilized directly by users. Its purpose
 * is to store essential details about a tensor's storage characteristics,
 * this allows operations that need tensor metadata but do not require
 * identification via names.
 *
 * TODO: rename to TensorMetadata?
 */
class UnnamedTensorInfo
{
public:
    UnnamedTensorInfo(DType dtype, const Shape& shape, SharedPtr<const Path> ptrPath, size_t begin, size_t end);

public:
    [[nodiscard]] DType dtype() const { return _dtype; }
    [[nodiscard]] const Shape& shape() const { return _shape; }
    [[nodiscard]] const Path& path() const { return *_ptrPath; }
    [[nodiscard]] size_t begin() const { return _begin; }
    [[nodiscard]] size_t end() const { return _end; }
    [[nodiscard]] size_t size() const { return _end - _begin; }
    [[nodiscard]] bool isStoredInFile() const { return _ptrPath != nullptr; }

public:
    [[nodiscard]] String to_string() const;


// IMPLEMENTATION
private:
    DType                 _dtype;    ///< The data type of the tensor
    Shape                 _shape;    ///< The shape of the tensor
    SharedPtr<const Path> _ptrPath;  ///< The file path or URL where the tensor is located (if any)
    size_t                _begin;    ///< The beginning of the tensor data in the source file (in bytes)
    size_t                _end;      ///< The end of the tensor data in the source file (in bytes)
};


}      // namespace tin
#endif // TIN_UNNAMEDTENSORINFO_H_
