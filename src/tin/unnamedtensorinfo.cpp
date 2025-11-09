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
 * @param dtype    The data type of the tensor elements (e.g., DType::F16)
 * @param shape    The shape of the tensor (e.g., Shape{128, 64})
 * @param ptrPath  The path to the file containing the tensor data.
 *                 (this path may be shared between multiple UnnamedTensorInfo objects)
 * @param begin    The offset in bytes from which the tensor data starts in the file.
 * @param end      The offset in bytes at which the tensor data ends in the file.
 */
UnnamedTensorInfo::UnnamedTensorInfo(DType                 dtype,
                                     const Shape&          shape,
                                     SharedPtr<const Path> ptrPath,
                                     size_t                begin,
                                     size_t                end
): _dtype  { dtype },
   _shape  { shape },
   _ptrPath{ ptrPath ? ptrPath : _ptrEmptyPath },
   _begin  { begin },
   _end    { end   }
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
        "    dtype: {},\n"\
        "    shape: {},\n"\
        "    path : \"{}\",\n"\
        "    begin: {},\n"\
        "    end  : {},\n"\
        "}}",
        _dtype,
        _shape.to_string(),
        _ptrPath ? _ptrPath->string() : "<nullptr>",
        _begin,
        _end
    );
}


} // namespace tin
