/*
| File    : metariant.cpp
| Purpose : Type-safe way to hold checkpoint metadata values.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 12, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <variant>     // for std::get, std::variant
#include <algorithm>   // for std::transform
#include <iterator>    // for std::back_inserter
#include <tin/common.h>
#include <tin/metavariant.h>
#include <tin/storagetype.h>
namespace tin {
using std::get;

// The 'MetaVariant::None' constant represents the absence of a meaningful value.
const MetaVariant MetaVariant::None = MetaVariant();

//================================ HELPERS ================================//

/**
 * Trims whitespace from both ends of a string and converts it to lowercase.
 *
 * This function removes any leading or trailing whitespace characters (spaces,
 * tabs, newlines, etc), and transforms the remaining characters into their
 * lowercase equivalents. The conversion assumes simple ASCII character
 * encoding for determining uppercase/lowercase.
 *
 * @param str The input string to process.
 * @return A string that is trimmed of whitespace from both ends and converted to lowercase.
 */
static std::string
_trim_to_lowercase(const std::string_view& str)
{
    static const auto whitespaces = std::string_view{" \t\n\r\f\v"};

    // identify the range of non-whitespace characters in the input string,
    // (if empty or all whitespace, return an empty string)
    const auto firstIdx = str.find_first_not_of(whitespaces);
    const auto lastIdx  = str.find_last_not_of(whitespaces);
    if( firstIdx >= lastIdx ) {
        return {};
    }

    // pre-allocate memory to enhance efficiency,
    // based on the length of the trimmed substring.
    std::string result;
    result.reserve( (lastIdx+1) - firstIdx ); 

    // transform the identified trimmed substring into lowercase
    // (while UTF-8 is compatible, only ASCII characters will be converted to lowercase)
    auto first = str.begin() + firstIdx;
    auto last  = str.begin() + lastIdx;
    std::transform(first, last, std::back_inserter(result),
                   [=](unsigned char ch) -> unsigned char
                   {
                       if     ( 'A' <= ch && ch <= 'Z' ) { return ch + ('a'-'A'); }
                       else if( ch  <  32              ) { return ' '; }
                       return ch;
                   });
    return result;
}    

//============================= CONSTRUCTION ==============================//

/**
 * Constructs a MetaVariant object representing a boolean value.
 *
 * This constructor initializes a MetaVariant to contain a boolean. The optional
 * `storageType` parameter serves only as a hint about the file storage format.
 *
 * @param value       The boolean value to be stored within the MetaVariant object.
 * @param storageType An optional hint specifying how the value is stored in file.
 *                    Default is `StorageType::BOOL`.
 */
MetaVariant::MetaVariant(bool        value,
                         StorageType storageType  // = StorageType::BOOL
) noexcept
: _type { Type::BOOL },
  _value{ value },
  _storageType{ storageType }
{}


/**
 * Constructs a MetaVariant object representing an integer value.
 *
 * This constructor initializes the MetaVariant to contain an integer. The `long`
 * type is used for accommodating values potentially larger than 32 bits. The
 * optional `storageType` parameter serves only as a hint about the file
 * storage format.
 *
 * @param value       The integer (long) value to be stored within the MetaVariant object.
 * @param storageType An optional hint specifying how the value is stored in file.
 *                    Default is `StorageType::INT32`.
 */
MetaVariant::MetaVariant(long        value,
                         StorageType storageType // = StorageType::INT32
) noexcept
: _type { Type::LONG_INT },
  _value{ value },
  _storageType{ storageType }
{}


/**
 * Constructs a MetaVariant object representing an unsigned integer value.
 *
 * This constructor initializes a MetaVariant to store an unsigned long integer.
 * The `unsigned long` type is used for accommodating values potentially larger
 * than 32 bits. The optional `storageType` parameter serves only as a hint
 * about the file storage format.
 *
 * @param value       The unsigned integer (long) value to be stored within
 *                    the MetaVariant object.
 * @param storageType An optional hint specifying how the value is stored in file.
 *                    Default is `StorageType::UINT32`.
 */
MetaVariant::MetaVariant(unsigned long value,
                         StorageType   storageType // = StorageType::UINT32
) noexcept
: _type { Type::LONG_UNSIGNED },
  _value{ value },
  _storageType{ storageType }
{}


/**
 * Constructs a MetaVariant object representing a double-precision floating-point value.
 *
 * This constructor initializes a MetaVariant to store a `double`. The optional
 * `storageType` parameter serves only as a hint about the file storage format.
 *
 * @param value       The double precision floating point value to be stored
 *                    within the MetaVariant object.
 * @param storageType An optional hint specifying how the value is stored in file.
 *                    Default is `StorageType::FLOAT32`.
 */
MetaVariant::MetaVariant(double      value,
                         StorageType storageType // = StorageType::FLOAT32
) noexcept
: _type { Type::DOUBLE },
  _value{ value },
  _storageType{ storageType }
{}


/**
 * Constructs a MetaVariant object representing a string.
 *
 * This constructor initializes a MetaVariant to store a `StringView`, which
 * represents a sequence of characters. The optional `storageType` parameter
 * serves only as a hint about the file storage format.
 *
 * @param value       The string to be stored within the MetaVariant object.
 * @param storageType An optional hint specifying how the value is stored in file.
 *                    Default is `StorageType::STRING`.
 */
MetaVariant::MetaVariant(StringView  value,
                         StorageType storageType // = StorageType::STRING
) noexcept
: _type { Type::STRING },
  _value{ String{value} },
  _storageType{ storageType }
{} 


/**
 * Private constructor used exclusively by `MetaVariant::None`.
 */
MetaVariant::MetaVariant() noexcept
: _type { Type::NONE }
{}
  

//============================== CONVERSIONS ==============================//

/**
 * Returns a boolean representation of this MetaVariant object.
 *
 * This function attempts to return the boolean value that best represents the
 * state of this MetaVariant object, regardless of its initial data type.
 * 
 * For string types, common truthy values such as "true", "yes", "on", "enabled",
 * and "1" are interpreted as `true`, while falsy values like "false", "no",
 * "off", "disabled", and "0" are interpreted as `false`.
 * 
 * For other types, a non-zero numeric value is interpreted as `true`, while
 * zero is interpreted as `false`.
 *
 * If the value connot be expressed as a boolean (such as empty fields or
 * malformed strings), the function returns the provided `default_` parameter.
 *
 * @param default_ An optional fallback value to be returned when boolean
 *                 conversion isn't possible. Default is false.
 * @return
 *     A boolean that represents this meta-variant.
 *     The return value is either the converted boolean, or the provided
 *     `default_` parameter upon failure.
 */
bool
MetaVariant::as_boolean(bool default_ // = false
) const noexcept {
    switch( _type )
    {
        case Type::BOOL:
            return get<bool>(_value);

        case Type::LONG_INT:
            return get<long int>(_value) != 0;

        case Type::LONG_UNSIGNED:
            return get<long unsigned>(_value) != 0;

        case Type::DOUBLE:
            return (get<double>(_value) < -0.1 || 0.1 < get<double>(_value));

        case Type::STRING: {
            String lower = _trim_to_lowercase( get<String>(_value) );
            if( lower == "true" || lower == "yes" || lower == "on" || lower == "enabled" || lower == "1" ) {
                return true;
            }
            if (lower == "false" || lower == "no" || lower == "off" || lower == "disabled" || lower == "0" ) {
                return false;
            }
            return default_;
            } break;

        default:
            return default_;
    }
}


/**
 * Returns a (long) integer representation of this MetaVariant object.
 *
 * This function attempts to return the (long) integer that best represents
 * this MetaVariant object. The `long` type was chosen for support values
 * potentially larger than 32 bits.
 *
 * It handles all different data types that MetaVariant can contain and manages
 * potential conversion issues. In cases of overflow, it returns the maximum
 * or minimum possible long value to provide a safe fallback.
 * 
 * If the value cannot be expressed as an integer (such as empty fields or
 * malformed strings), the function returns the `default_` parameter.
 *
 * @param default_ An optional fallback value to be returned when integer
 *                 representation isn't possible. Default is 0L.
 * @return
 *     A (long) integer that represents this meta-variant.
 *     The return value is either the converted integer, max/min long in case
 *     of overflow, or the provided `default_` parameter upon failure.
 */
long
MetaVariant::as_integer(long default_ // = 0L
) const noexcept {
    switch( _type )
    {
        case Type::BOOL:
            return get<bool>(_value) ? 1L : 0L;

        case Type::LONG_INT:
            return get<long int>(_value);

        case Type::LONG_UNSIGNED:
            // solve potential overflow when converting
            if( get<long unsigned>(_value) > std::numeric_limits<long>::max() )
            { return std::numeric_limits<long>::max(); }
            return static_cast<long>( get<long unsigned>(_value) );
        
        case Type::DOUBLE:
            // solve potential overflow when converting
            if( get<double>(_value) > std::numeric_limits<long>::max() )
            { return std::numeric_limits<long>::max(); }
            if ( get<double>(_value) < std::numeric_limits<long>::min() )
            { return std::numeric_limits<long>::min();  }
            return static_cast<long>( get<double>(_value) );

        case Type::STRING:
            try {
                return std::stol( get<String>(_value) );
            } catch (...) { return default_; }

        default:
            return default_;
    }
}


/**
 * Returns an unsigned (long) integer representation of this MetaVariant object.
 *
 * This function attempts to return the unsigned (long) integer that best
 * represents this MetaVariant object. The `unsigned long` type was chosen to
 * support values potentially larger than 32 bits.
 *
 * It handles all different data types that MetaVariant can contain and manages
 * potential conversion issues. In cases of overflow, it returns the maximum
 * possible unsigned long value to provide a safe fallback.
 * 
 * If the value cannot be expressed as an unsigned integer (such as empty
 * fields, or malformed strings), the function returns the `default_`
 * parameter.
 *
 * @param default_ An optional fallback value to be returned when unsigned
 *                 integer representation isn't possible. Default is 0UL.
 * @return
 *     An unsigned (long) integer that represents this meta-variant.
 *     The return value is either the converted unsigned integer, max/min
 *     unsigned long value in case of overflow, or the provided `default_`
 *     parameter upon failure.
 */
unsigned long
MetaVariant::as_unsigned(unsigned long default_ // = 0UL
) const noexcept {
    switch( _type ) {

        case Type::BOOL:
            return get<bool>(_value) ? 1UL : 0UL;

        case Type::LONG_INT:
            // solve potential negative values
            if( get<long int>(_value) < 0 ) { return 0UL; }
            return static_cast<unsigned long>( get<long int>(_value) );

        case Type::LONG_UNSIGNED:
            return get<long unsigned>(_value);

        case Type::DOUBLE:
            // solve potential overflow when converting
            if( get<double>(_value)> std::numeric_limits<unsigned long>::max() )
            { return std::numeric_limits<unsigned long>::max(); }
            if ( get<double>(_value) < std::numeric_limits<unsigned long>::min() )
            { return std::numeric_limits<unsigned long>::min();  }
            return static_cast<unsigned long>( get<double>(_value) );

        case Type::STRING: {
            try {
                return std::stoul( get<String>(_value) );
            } catch (...) { return default_; }
        }
        default:
            return default_;
    }
}


/**
 * Returns a double precision floating point representation of this MetaVariant object.
 *
 * This function attempts to return the double precision floating point value
 * that best represents this MetaVariant object. It handles all different data
 * types that MetaVariant can contain and manages potential conversion issues.
 *
 * For numeric types, the conversion is straightforward. For boolean types, true
 * is converted to 1.0 and false to 0.0. For string types, the standard string
 * to double conversion is used.
 *
 * If the value cannot be expressed as a double (such as empty fields or
 * malformed strings), the function returns the provided `default_` parameter.
 *
 * @param default_ An optional fallback value to be returned when double
 *                 representation isn't possible. Default is 0.0.
 * @return
 *     A double precision floating point value that represents this meta-variant.
 *     The return value is either the converted double, or the provided
 *     `default_` parameter upon failure.
 */
double
MetaVariant::as_double(double default_ // = 0.0
) const noexcept {
    switch( _type ) {

        case Type::BOOL:
            return get<bool>(_value) ? 1.0 : 0.0;

        case Type::LONG_INT:
            return static_cast<double>( get<long int>(_value) );

        case Type::LONG_UNSIGNED:
            return static_cast<double>( get<long unsigned>(_value) );

        case Type::DOUBLE:
            return get<double>(_value);

        case Type::STRING:
            try {
                return std::stod( get<String>(_value) );
            } catch (...) { return default_; }

        default:
            return default_;
    }
}


/**
 * Returns a string representation of this MetaVariant object.
 *
 * This function attempts to return a string representation of this MetaVariant
 * object. It handles all different data types that MetaVariant can contain and
 * manages potential conversion issues.
 * 
 * For numeric types, the standard string representation is used, while for
 * boolean types, "true" or "false" is returned.
 *
 * If the value cannot be converted to a string (such as empty fields), the
 * function returns the provided `default_` parameter.
 *
 * @param default_  An optional fallback text to be returned when string 
 *                  representation isn't possible. Default is an empty string.
 * @return
 *     A string that represents this meta-variant.
 *     The return value is either the converted string, or the provided
 *     `default_` parameter upon failure.
 */
String
MetaVariant::as_string(StringView default_ // = ""
) const noexcept {
    switch( _type ) {
        case Type::BOOL         : return get<bool>(_value) ? "true" : "false";
        case Type::LONG_INT     : return std::to_string( get<long int     >(_value) );
        case Type::LONG_UNSIGNED: return std::to_string( get<long unsigned>(_value) );
        case Type::DOUBLE       : return std::to_string( get<double       >(_value) );
        case Type::STRING       : return get<String>(_value);
        default:
            return String{ default_ };
    }
}


} // namespace tin
