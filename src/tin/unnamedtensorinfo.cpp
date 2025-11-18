/*
| File    : unnamedtensorinfo.cpp
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
#include <format> // for std::format
#include <tin/unnamedtensorinfo.h>
namespace tin {

static SharedPtr<const Path> _ptrEmptyPath = std::make_shared<Path>();

//============================= CONSTRUCTION ==============================//

/**
 * Constructors a UnnamedTensorInfo with all its attributes.
 *
 * @param dtype        The data type of the tensor elements (e.g., DType::F16)
 * @param shape        The shape of the tensor (e.g., Shape{128, 64})
 * @param ptrPath      The path to the file containing the tensor data.
 *                     (this path may be shared between multiple UnnamedTensorInfo objects)
 * @param rawDataBegin The offset in bytes from which the tensor data starts in the file.
 * @param rawDataEnd   The offset in bytes at which the tensor data ends in the file.
 */
UnnamedTensorInfo::UnnamedTensorInfo(DType                 dtype,
                                     const Shape&          shape,
                                     SharedPtr<const Path> ptrPath,
                                     std::streampos        rawDataBegin,
                                     std::streampos        rawDataEnd
) noexcept
: _dtype       { dtype },
  _shape       { shape },
  _ptrPath     { ptrPath && !ptrPath->empty() ? ptrPath : _ptrEmptyPath },
  _rawDataBegin{ rawDataBegin },
  _rawDataEnd  { rawDataEnd   }
{}

/**
 * Move constructor with optional offset for `rawDataBegin` and `rawDataEnd`.
 *
 * This move constructor initializes an `UnnamedTensorInfo` object by moving
 * data from other instance to this, with the option to apply an offset to the
 * properties `rawDataBegin` and `rawDataEnd`. If no offset is specified, it
 * defaults to zero, effectively performing a standard move operation.
 * 
 * @param other         The UnnamedTensorInfo object being moved from.
 * @param rawDataOffset An optional offset for `rawDataBegin` and `rawDataEnd`.
 */
UnnamedTensorInfo::UnnamedTensorInfo(UnnamedTensorInfo&& other,
                                     std::streampos      rawDataOffset // = 0
) noexcept
: _dtype       { std::move(other._dtype)   },
  _shape       { std::move(other._shape)   },
  _ptrPath     { std::move(other._ptrPath) },
  _rawDataBegin{ other._rawDataBegin + rawDataOffset },
  _rawDataEnd  { other._rawDataEnd   + rawDataOffset }
{}

/**
 * Copy constructor with optional offset for `rawDataBegin` and `rawDataEnd`.
 *
 * This copy constructor creates a new `UnnamedTensorInfo` object as a copy
 * of an existing one, with the option to apply an offset to the properties
 * `rawDataBegin` and `rawDataEnd`. If no offset is specified, it defaults to
 * zero, resulting in a standard copy operation.
 *
 * @param other         The UnnamedTensorInfo object being copied from.
 * @param rawDataOffset An optional offset for `_rawDataBegin` and `_rawDataEnd`.
 */
UnnamedTensorInfo::UnnamedTensorInfo(const UnnamedTensorInfo& other,
                                     std::streampos           rawDataOffset // = 0
) noexcept
: _dtype       { other._dtype   },
  _shape       { other._shape   },
  _ptrPath     { other._ptrPath },
  _rawDataBegin{ other._rawDataBegin + rawDataOffset },
  _rawDataEnd  { other._rawDataEnd   + rawDataOffset }
{}

//=============================== DEBUGGING ===============================//

/**
 * Returns a string representation of the object for debugging purposes.
 * 
 * This function creates a human-readable string representation of the 
 * object internal state, which can be useful for debugging and logging.
 * @return A string that provides a clear representation of the object.
 */
String
UnnamedTensorInfo::to_string() const {
    return std::format(
        "UnnamedTensorInfo{{\n"\
        "    _dtype       : {},  \n"\
        "    _shape       : {},  \n"\
        "    _ptrPath     : '{}',\n"\
        "    _rawDataBegin: {},  \n"\
        "    _rawDataEnd  : {},  \n"\
        "}}",
        _dtype,
        _shape,
        _ptrPath ? _ptrPath->string() : "<nullptr>",
        static_cast<size_t>(_rawDataBegin),
        static_cast<size_t>(_rawDataEnd)
    );
}


} // namespace tin
