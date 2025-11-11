/*
| File    : tensorinfo.cpp
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
#include <format>
#include <tin/tensorinfo.h>
#include <tin/unnamedtensorinfo.h>
namespace tin {

//======================= CONSTRUCTION/DESTRUCTION ========================//

/**
 * Constructors a TensorInfo with all its attributes.
 *
 * @param name         The name of the tensor (e.g., "encoder.block.0.layer.0.SelfAttention.k.weight")
 * @param dtype        The data type of the tensor elements (e.g., DType::F16)
 * @param shape        The shape of the tensor (e.g., Shape{128, 64})
 * @param path         The path to the file containing the tensor data.
 * @param rawDataBegin The offset in bytes from which the tensor data starts in the file.
 * @param rawDataEnd   The offset in bytes at which the tensor data ends in the file.
 */
TensorInfo::TensorInfo(StringView     name,
                       DType          dtype,
                       const Shape&   shape,
                       StringView     path,
                       std::streampos rawDataBegin,
                       std::streampos rawDataEnd
):
TensorInfo(
    make<String>(name),
    make<UnnamedTensorInfo>( dtype, shape, make<Path>(path), rawDataBegin, rawDataEnd )
)
{}


/**
 * Constructors a TensorInfo with all its attributes.
 *
 * This constructor is designed to allow the sharing of the `path` attribute 
 * among multiple TensorInfo instances. It is crucial that the provided path
 * remains constant since TensorInfo objects are immutable, necessitating
 * careful usage.
 *
 * @param name         The name of the tensor (e.g., "encoder.block.0.layer.0.SelfAttention.k.weight")
 * @param dtype        The data type of the tensor elements (e.g., DType::F16)
 * @param shape        The shape of the tensor (e.g., Shape{128, 64})
 * @param ptrPath      The path to the file containing the tensor data.
 *                     (this path may be shared between multiple TensorInfo instances)
 * @param rawDataBegin The offset in bytes from which the tensor data starts in the file.
 * @param rawDataEnd   The offset in bytes at which the tensor data ends in the file.
 * 
 * Note: Due to immutability constraints, ensure that any shared paths are not
 * altered after TensorInfo objects have been constructed with them.
 */
TensorInfo::TensorInfo(StringView            name,
                       DType                 dtype,
                       const Shape&          shape,
                       SharedPtr<const Path> ptrPath,
                       std::streampos        rawDataBegin,
                       std::streampos        rawDataEnd
):
TensorInfo(
    make<String>(name),
    make<UnnamedTensorInfo>( dtype, shape, ptrPath, rawDataBegin, rawDataEnd )
)
{}


/**
 * Constructs a TensorInfo object with a given name and an existing tensor info.
 *
 * This constructor initializes a TensorInfo object using the provided tensor's
 * name and an existing TensorInfo object. It copies the internal tensor
 * information from the other TensorInfo object and assigns the newly provided
 * name. This is useful when you want to create a new TensorInfo with a
 * different name while preserving its original tensor metadata.
 *
 * @param name  The desired name for the tensor.
 * @param other An existing TensorInfo instance whose information will be copied.
 */
TensorInfo::TensorInfo(StringView        name,
                       const TensorInfo& other
):
TensorInfo(
    make<String>(name),
    other._ptrUnnamedTensorInfo
)
{}


/**
 * Private constructor for internal use by public constructors.
 *
 * This private constructor initializes a TensorInfo object using shared
 * pointers. It is intended for use by the public constructors to encapsulate
 * common initialization logic.
 *
 * @param ptrName              A SharedPtr to a String representing the tensor's name.
 * @param ptrUnnamedTensorInfo A SharedPtr to an UnnamedTensorInfo object containing tensor metadata.
 */
TensorInfo::TensorInfo(SharedPtr<String> ptrName,
                       SharedPtr<UnnamedTensorInfo> ptrUnnamedTensorInfo
): _ptrName{ ptrName },
   _ptrUnnamedTensorInfo{ ptrUnnamedTensorInfo }
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
TensorInfo::to_string() const {
    return std::format(
        "UnnamedTensorInfo{{\n"\
        "    _ptrName : '{}',\n"\
        "    _ptrUnnamedTensorInfo: {}\n"\
        "}}",
        _ptrName              ? *_ptrName                           : "<nullptr>",
        _ptrUnnamedTensorInfo ?  _ptrUnnamedTensorInfo->to_string() : "<nullptr>"
    );
}

} // namespace tin
