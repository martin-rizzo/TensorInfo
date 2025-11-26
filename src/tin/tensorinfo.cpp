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

// The 'TensorInfo::None' constant represents the absence of a tensor info.
const TensorInfo TensorInfo::None = TensorInfo();

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
) noexcept
: _ptrName{ make<String>(name) },
  _ptrUnnamedTensorInfo{ other._ptrUnnamedTensorInfo }
{}

/**
 * Move constructor with optional offset for `rawDataBegin` and `rawDataEnd`.
 *
 * This constructor initializes a `TensorInfo` object by moving from another 
 * `TensorInfo` instance. It allows an optional offset to be applied to the 
 * properties `rawDataBegin` and `rawDataEnd`. If no offset is specified, it
 * defaults to zero, effectively performing a move without modification.
 *
 * @param other         The TensorInfo object being moved from.
 * @param rawDataOffset An optional offset for `rawDataBegin` and `rawDataEnd`.
 */
TensorInfo::TensorInfo(TensorInfo&&   other,
                       std::streampos rawDataOffset // = 0
) noexcept
: _ptrName{ std::move(other._ptrName) },
  _ptrUnnamedTensorInfo{
      rawDataOffset == 0
          ? std::move(other._ptrUnnamedTensorInfo)
          : std::make_shared<UnnamedTensorInfo>( std::move(*other._ptrUnnamedTensorInfo), rawDataOffset )
  }
{}

/**
 * Copy constructor with optional offset for `rawDataBegin` and `rawDataEnd`.
 *
 * This constructor creates a new `TensorInfo` object as a copy of an existing 
 * one, with the option to apply an offset to the properties `rawDataBegin` and
 * `rawDataEnd`. If no offset is specified, it defaults to zero, resulting in
 * a direct copy.
 *
 * @param other         The TensorInfo object being copied from.
 * @param rawDataOffset An optional offset for `rawDataBegin` and `rawDataEnd`.
 */
TensorInfo::TensorInfo(const TensorInfo& other,
                       std::streampos    rawDataOffset // = 0
) noexcept
: _ptrName{ other._ptrName },
  _ptrUnnamedTensorInfo{
      rawDataOffset == 0
          ? other._ptrUnnamedTensorInfo
          : std::make_shared<UnnamedTensorInfo>( *other._ptrUnnamedTensorInfo, rawDataOffset )
  }
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
TensorInfo::TensorInfo(SharedPtr<String>            ptrName,
                       SharedPtr<UnnamedTensorInfo> ptrUnnamedTensorInfo
) noexcept
: _ptrName{ ptrName },
  _ptrUnnamedTensorInfo{ ptrUnnamedTensorInfo }
{}


/**
 * Private constructor used exclusively by `TensorInfo::None`.
 */
TensorInfo::TensorInfo() noexcept
: _ptrName{ nullptr },
  _ptrUnnamedTensorInfo{ make<UnnamedTensorInfo>( DType::UNKNOWN, Shape{}, make<Path>(""), 0, 0 ) }
{}

//============================== ATTRIBUTES ===============================//

/**
 * Returns name of the tensor.
 */
const String&
TensorInfo::name() const noexcept {
    static const String EmptyString{};
    return _ptrName ? *_ptrName : EmptyString;
}

/**
 * Returns the name of this tensor relative to a given prefix.
 *
 * This function takes a prefix and removes it from the tensor's full name,
 * returning the remaining portion as a StringView.
 * 
 * If the prefix is not valid (i.e., it doesn't match at the beginning of the
 * tensor's name or doesn't cover a complete segment), an empty string is
 * returned.
 *
 * @param prefix The prefix to remove from the tensor's full name.
 *               (for example: "model.layers.25.self_attn")
 * @return A string view containing the relative name after removing the prefix, 
 *         or an empty string if the prefix is invalid.
 */
StringView
TensorInfo::relative_name(StringView prefix) const noexcept {
    // if no prefix was provided, return the full name
    if( prefix.empty() ) { return name(); }

    // remove any trailing dot from the user provided prefix
    if( prefix.ends_with('.') ) { prefix.remove_suffix(1); }

    // if the prefix does not match at position 0, it is invalid
    auto pos = name().find(prefix);
    if( pos != 0 ) { return ""; }

    // create a string_view from the node's name and remove the prefix
    StringView result{ name() };
    result.remove_prefix( prefix.size() );

    // if the resulting name does not start with a dot,
    // the prefix is invalid because it does not cover a complete segment
    if( !result.starts_with('.') ) {
        return "";
    }
    // return the remaining relative name after removing the initial dot
    result.remove_prefix(1);
    return result;
}

/**
 * Generates a lexicographically sortable name.
 *
 * This function processes the tensor's name to ensure correct lexicographic
 * sorting. Among other things, it replaces sequences of digits with zero-padded
 * versions, ensuring that numeric values within the name are ordered correctly.
 *
 * @note This operation can be computationally expensive. If used in loops,
 * it's recommended to compute this function once and store its result for reuse.
 *
 * @return
 *   A string representing the normalized name suitable for lexicographic sorting.
 */
String TensorInfo::generate_normalized_name() const noexcept {
    const auto input   = name();
    const auto length  = name().length();
    String     result;
    result.reserve(length+32);

    size_t start = 0;
    for( size_t i = start ; i<length ; /* 'i' is incremented inside loop */ )
    {
        // skip all non-digit characters
        while( i<length && !isdigit(input[i]) ) { ++i; }

        // at this point, either the end of the string is reached
        // or a digit character follows
        if( i<length ) {

            // avanzamos j para encontrar el final de la secuencia de digitos
            size_t j = i+1; while( j<length && isdigit(input[j]) ) { ++j; }

            // if the digit sequence length is <= 8 characters
            // then replace it with zero-padded version
            if( (j-i) < 8 ) {
                result += input.substr(start, i-start) + std::string(8-(j-i),'0') + input.substr(i,j-i);
            } else {
                result += input.substr(start, j-start);
            }
            start = i = j;
        }
    }
    // append any remaining characters at the end
    // of the string if not copied yet
    if( start < length ) { result += input.substr(start); }
    return result;
}

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
