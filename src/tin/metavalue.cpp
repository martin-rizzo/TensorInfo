/*
| File    : metavalue.cpp
| Purpose : Each value stored in the metadata information of the checkpoint.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 12, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <algorithm>   // for std::transform, std::find_if
#include <iterator>    // for std::back_inserter, std::distance
#include <tin/common.h>
#include <tin/metavalue.h>
#include <tin/metatype.h>
namespace tin {

// The 'MetaValue::None' constant represents the absence of a meaningful value.
const MetaValue MetaValue::None = MetaValue();

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

    // obtener la subcadena trimeada (first index, last index)
    // (return an empty string if all characters are whitespace)
    const auto firstIdx = str.find_first_not_of(whitespaces);
    const auto lastIdx  = str.find_last_not_of(whitespaces);
    if( firstIdx >= lastIdx ) {
        return {};
    }

    // pre-allocate memory for efficiency
    std::string result;
    result.reserve( (lastIdx+1) - firstIdx ); 

    // transform the trimmed substring to lowercase,
    // replacing non-printable characters with spaces
    // (assuming ASCII encoding, utf8 can work but only ascii caracters will be converted)
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

MetaValue::MetaValue(bool     value,
                     MetaType metatype  // = MetaType::BOOL
) noexcept
: _type { Type::BOOLEAN    },
  _value{ .boolean = value },
  _metatype{ metatype }
{}


MetaValue::MetaValue(long int  value,
                     MetaType  metatype // = MetaType::INT32
) noexcept
: _type { Type::LONG_INT    },
  _value{ .long_int = value },
  _metatype{ metatype }
{}

MetaValue::MetaValue(long unsigned value,
                     MetaType      metatype // = MetaType::UINT32
) noexcept
: _type { Type::LONG_UNSIGNED    },
  _value{ .long_unsigned = value },
  _metatype{ metatype }
{}

MetaValue::MetaValue(double   value,
                     MetaType metatype // = MetaType::FLOAT32
) noexcept
: _type { Type::DOUBLE_PRECISION    },
  _value{ .double_precision = value },
  _metatype{ metatype }
{}


MetaValue::MetaValue(StringView value,
                     MetaType metatype // = MetaType::STRING
) noexcept
: _type        { Type::STRING },
  _value_string{    value     },
  _metatype{ metatype }
{} 

// private, utilizada unicamente por `MetaValue::None`
MetaValue::MetaValue() noexcept
: _type { Type::NONE }
{}
  

//============================== CONVERSIONS ==============================//

bool
MetaValue::as_boolean(bool default_ // = false
) const noexcept {
    switch( _type ) {
        case Type::BOOLEAN          : return _value.boolean;
        case Type::LONG_INT         : return (_value.long_int      != 0);
        case Type::LONG_UNSIGNED    : return (_value.long_unsigned != 0);
        case Type::DOUBLE_PRECISION : return (_value.double_precision < -0.1 || 0.1 < _value.double_precision);
        case Type::STRING: {
            String lower = _trim_to_lowercase( _value_string );
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

String
MetaValue::as_string(StringView default_ // = ""
) const noexcept {
    switch( _type ) {
        case Type::BOOLEAN         : return _value.boolean ? "true" : "false";
        case Type::LONG_INT        : return std::to_string(_value.long_int);
        case Type::LONG_UNSIGNED   : return std::to_string(_value.long_unsigned);
        case Type::DOUBLE_PRECISION: return std::to_string(_value.double_precision);
        case Type::STRING          : return _value_string;
        default:
            return String{ default_ };
    }
}


} // namespace tin
