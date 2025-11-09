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
