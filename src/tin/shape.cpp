/*
| File    : shape.cpp
| Purpose : A class representing the shape of a tensor.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 8, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <sstream> // for std::stringstream
#include <tin/shape.h>
namespace tin {

/**
 * Returns a string representation of the object for debugging purposes.
 *
 * This function generates a human-readable string that represents 
 * the dimensions stored in the `Shape` object. It is primarily intended 
 * for use during debugging and error logging to easily understand or 
 * display the shape's structure.
 *
 * @return A string that provides a clear representation of the `Shape` object.
 */
String
Shape::to_string() const {
    std::ostringstream os;

    // wrap the dims in square brackets
    os << "[";
    bool first = true;
    for( auto dim : _dims ) {
        if (first) { os << dim; first=false; }
        else       { os << ", " << dim; }
    }
    os << "]";

    // return the generated string
    return os.str();
}


}


