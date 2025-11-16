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


/**
 * Store all metadata about a tensor.
 *
 * The TensorInfo class acts as an interface to access various attributes of
 * a tensor such as its name, shape, data type, and offsets within the file.
 * Internally, it uses shared pointers for managing dynamic resources
 * efficiently.
 */
class TensorInfo
{
// CONSTANT VALUES
public:
    static const TensorInfo None;

// CONSTRUCTION/DESTRUCTION
public:
    TensorInfo(StringView name, DType dtype, const Shape& shape, StringView            path   , std::streampos rawDataBegin, std::streampos rawDataEnd);
    TensorInfo(StringView name, DType dtype, const Shape& shape, SharedPtr<const Path> ptrPath, std::streampos rawDataBegin, std::streampos rawDataEnd);
    TensorInfo(StringView name, const TensorInfo& other);
    TensorInfo(const TensorInfo& other) = default;
    TensorInfo(TensorInfo&& other) = default;
    ~TensorInfo() = default;
private:
    TensorInfo(SharedPtr<String> ptrName, SharedPtr<UnnamedTensorInfo> ptrUnnamedTensorInfo);
    TensorInfo();

// ATTRIBUTES
public:
    [[nodiscard]] const String&      name()           const noexcept;
    [[nodiscard]] DType              dtype()          const noexcept;
    [[nodiscard]] const Shape&       shape()          const noexcept;
    [[nodiscard]] unsigned long long numel()          const noexcept;
    [[nodiscard]] const Path&        path()           const noexcept;
    [[nodiscard]] std::streampos     raw_data_begin() const noexcept;
    [[nodiscard]] std::streampos     raw_data_end()   const noexcept;
    [[nodiscard]] std::streamsize    raw_data_size()  const noexcept;
    [[nodiscard]] String generate_normalized_name() const noexcept;
    [[nodiscard]] bool is_file_stored() const;

// DEBUGGING
public:
    [[nodiscard]] String to_string() const;

// IMPLEMENTATION
private:
    SharedPtr<String>            _ptrName;               ///< A pointer to the name of this tensor
    SharedPtr<UnnamedTensorInfo> _ptrUnnamedTensorInfo;  ///< A pointer to an unnamed tensor info
};

//============================== ATTRIBUTES ===============================//

/**
 * Returns the data type of the tensor elements.
 * @return A DType enum value representing the data type of this tensor.
 */
inline DType
TensorInfo::dtype() const noexcept {
    return _ptrUnnamedTensorInfo->dtype();
}

/**
 * Returns shape of the tensor.
 * @return A Shape object containing the dimensions of this tensor.
 */
inline const Shape&
TensorInfo::shape() const noexcept {
    return _ptrUnnamedTensorInfo->shape();
}

/**
 * Returns number of elements in the tensor.
 * @return The total number of elements in this tensor.
 */
inline unsigned long long
TensorInfo::numel() const noexcept {
    return shape().numel();
}

/**
 * Returns path to the file where this tensor is stored.
 * @return A Path object representing the path.
 */
inline const Path&
TensorInfo::path() const noexcept {
    return _ptrUnnamedTensorInfo->path();
}

/**
 * Returns the offset in bytes from the beginning of a tensor's raw data within the file.
 */
inline std::streampos
TensorInfo::raw_data_begin() const noexcept {
    return _ptrUnnamedTensorInfo->raw_data_begin();
}

/**
 * Returns the offset in bytes at which the tensor's raw data ends within the file.
 */
inline std::streampos
TensorInfo::raw_data_end() const noexcept {
    return _ptrUnnamedTensorInfo->raw_data_end();
}

/**
 * Returns the size, in bytes, of the tensor's raw data.
 */
inline std::streamsize
TensorInfo::raw_data_size() const noexcept {
    return _ptrUnnamedTensorInfo->raw_data_size();
}

/**
 * Returns true if this tensor is stored in a file, false otherwise.
 * @details To access the file containing the tensor's raw data, use the path() method.
 */
inline bool
TensorInfo::is_file_stored() const {
    return _ptrUnnamedTensorInfo->is_file_stored();
}

} // namespace tin
//============================= PRINT SUPPORT =============================//

/**
 * Overloads the insertion (<<) operator for outputting TensorInfo objects to streams.
 *
 * This function allows `TensorInfo` objects to be directly inserted into
 * output streams such as `std::cout`, generating a human-readable
 * representation of the internal state of the object. It is primarily
 * intended for use in debugging and logging scenarios.
 *
 * @param os   The output stream where the TensorInfo will be written.
 * @param type The TensorInfo object to be inserted into the output stream.
 * @return A reference to 'os' for chaining.
 */
inline 
std::ostream& operator<<(std::ostream& os, const tin::TensorInfo& tensorInfo) {
    return os << tensorInfo.to_string();
}

/**
 * Formatter specialization for TensorInfo objects.
 * 
 * Enables using C++20's formatting framework with TensorInfo objects. The
 * formatter converts the object to its human-readable representation and
 * formats it accordingly. It is primarily intended for use in debugging and
 * logging scenarios.
 */
template <>
struct std::formatter<tin::TensorInfo> : std::formatter<std::string_view>
{
    auto format(const tin::TensorInfo& tensorinfo, format_context& ctx) const {
        return std::formatter<std::string_view>::format(tensorinfo.to_string(), ctx);
    }
};


#endif // TIN_TENSORINFO_H_
