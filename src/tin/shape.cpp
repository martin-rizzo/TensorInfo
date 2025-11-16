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

//=============================== DEBUGGING ===============================//

/**
 * Returns a string representation of the object for debugging purposes.
 *
 * This function generates a human-readable string that represents 
 * the dimensions stored in the `Shape` object. It is primarily intended 
 * for use during debugging and error logging to easily understand or 
 * display the shape's structure.
 *
 * @param brackets
 *    Optional string containing the opening and closing brackets used to wrap
 *    the dimension values. The first character is used as the opening bracket
 *    and the second as the closing bracket.
 * @param separator
 *    Optional string used to separate individual dimension values in the
 *    output. If not provided, a comma followed by a space is used as default.
 * @return
 *    A string providing a human-readable representation of the `Shape` object.
 */
String
Shape::to_string(StringView brackets, // = "[]",
                 StringView separator // = ", "
) const {
    std::ostringstream os;
    bool hasBrackets = brackets.size() >= 2;

    
    if( hasBrackets ) { os << brackets[0]; }
    bool first = true;
    for( auto dim : _dims ) {
        if( first ) { first=false; } else { os << separator; }
        os << dim;
    }
    if( hasBrackets ) { os << brackets[1]; }

    // return the generated string
    return os.str();
}


}


