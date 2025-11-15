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
#include <tin/common.h>
#include <tin/storagetype.h>
namespace tin {


class MetaValue
{
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
    bool           as_boolean (bool          default_ = false) const noexcept;
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

public:
    static const MetaValue None;   

// IMPLEMENTATION
private:
    enum class Type {
        NONE,
        BOOLEAN,
        LONG_INT,
        LONG_UNSIGNED,
        DOUBLE_PRECISION,
        STRING
        //STRING_VECTOR,
        //METAVALUE_VECTOR
    };
private:
    typedef union {
        bool           boolean;
        long int       long_int;
        long unsigned  long_unsigned;
        double         double_precision;
    } ValueUnion;
private:
    Type           _type;
    ValueUnion     _value;
    String         _value_string;
    StorageType    _storageType;
};


//============================= CONSTRUCTION ==============================//

inline
MetaValue::MetaValue(MetaValue&& other) noexcept
: _type { other._type  },
  _value{ other._value },
  _value_string{ std::move(other._value_string) },
  _storageType{ other._storageType }
{
    other._type = Type::NONE;
}


}      // namespace tin
#endif // TIN_METAVALUE_H_
