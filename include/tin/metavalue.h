/*
| File    : metavalue.h
| Purpose : Each value stored in the metadata information of the checkpoint.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 12, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_METAVALUE_H_
#define TIN_METAVALUE_H_
#include <variant> // for std::variant
#include <tin/common.h>
#include <tin/storagetype.h>
namespace tin {


class MetaValue
{
// CONSTANT VALUES
public:
    static const MetaValue None;   

// CONSTRUCTION
public:
    explicit MetaValue(bool           value , StorageType storageType = StorageType::BOOL   ) noexcept;
    explicit MetaValue(long           value , StorageType storageType = StorageType::INT32  ) noexcept;
    explicit MetaValue(unsigned long  value , StorageType storageType = StorageType::UINT32 ) noexcept;
    explicit MetaValue(double         value , StorageType storageType = StorageType::FLOAT32) noexcept;
    explicit MetaValue(StringView     value , StorageType storageType = StorageType::STRING ) noexcept;
    //explicit MetaValue(Vector<String> vector, StorageType storageType = StorageType::ARRAY_OF_STRINGS ) noexcept;
    explicit MetaValue(MetaValue&& other) noexcept;
private:
    MetaValue() noexcept;


// CONVERSIONS
public:
    bool           as_bool    (bool          default_ = false) const noexcept;
    long           as_integer (long          default_ = 0L   ) const noexcept;
    unsigned long  as_unsigned(unsigned long default_ = 0UL  ) const noexcept;
    double         as_double  (double        default_ = 0.0  ) const noexcept;
    String         as_string  (StringView    default_ = ""   ) const noexcept;
    //Vector<String> as_string_vector() const noexcept;
    
// TYPE HINTS
public:
    StorageType storage_type() const {
        return _storageType;
    }

// IMPLEMENTATION
private:
    using      Variant = std::variant<bool, long int, unsigned long, double, String>;
    enum class Type { NONE, BOOL, LONG_INT, LONG_UNSIGNED, DOUBLE, STRING
                      /*STRING_VECTOR, METAVALUE_VECTOR*/ };
private:
    Type        _type;
    Variant     _value;
    StorageType _storageType;
};


//============================= CONSTRUCTION ==============================//

inline
MetaValue::MetaValue(MetaValue&& other) noexcept
: _type { other._type  },
  _value{ other._value },
  _storageType{ other._storageType }
{
    other._type = Type::NONE;
}


}      // namespace tin
#endif // TIN_METAVALUE_H_
