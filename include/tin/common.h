/*
| File    : <tin/common.h>
| Purpose : Common types used throughout the "TensorInfo" library.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_COMMON_H_
#define TIN_COMMON_H_
#include <memory>       // for std::shared_ptr
#include <string>       // for std::string
#include <filesystem>   // for std::filesystem::path
#include <vector>       // for std::vector
#include <unordered_map>// for std::unordered_map
namespace tin {

using DataPos = size_t;  ///< Position of data within a file or stream (bytes)

/*================================= SIMPLE-CODE ALIASES ==================================================
  A set of templates to simplify code's visual appearance (?)
 +------------------+-------------------------------------------------------+-----------------------------+
 | SIMPLE-CODE      | Description                                           | Implemented using           |
 | ---------------- | ----------------------------------------------------- | --------------------------- |
 | SharedPtr<T>     | Manages shared ownership of a `T` object.             | `std::shared_ptr<T>`        |
 | Vector<T>        | A sequence of objects of type `T`.                    | `std::vector<T>`            |
 | Map<T>           | An associative container that maps keys to values.    | `std::unordered_map<T>`     |
 | String           | A string of characters.                               | `std::string`               |
 | StringView       | A lightweight view of a string.                       | `std::string_view`          |
 | Path             | A path in the filesystem.                             | `std::filesystem::path`     |
 | make_shared<T>() | Constructs a `T` object and returns a `SharedPtr<T>`. | `std::make_shared<T>()`     |
*/

/** A shared pointer to an object of type T. (implemented using std::shared_ptr<T>) */
template <typename T>
using SharedPtr = std::shared_ptr<T>;

/** A vector of objects of type T. (implemented using std::vector<T>) */
template <typename T>
using Vector = std::vector<T>;

/** A map of objects of type T. (implemented using std::unordered_map<T>) */
template <typename Key, typename Value>
using Map = std::unordered_map<Key, Value>;

/** A sequence of characters. */
using String = std::string;

/** A lightweight view of a string. (implemented using std::string_view) */
using StringView = std::string_view;

/** A path in the filesystem. */
using Path = std::filesystem::path;

/** Creates an object of type T using the provided arguments and wraps it in Ptr<T>. (implemented using std::make_unique<T>) */
template<typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}


}      // namespace tin
#endif // TIN_COMMON_H_
