/*
| File    : metavariant.h
| Purpose : Type-safe way to hold checkpoint metadata values.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 12, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_METAVARIANT_H_
#define TIN_METAVARIANT_H_
#include <variant> // for std::variant
#include <tin/common.h>
#include <tin/storagetype.h>
namespace tin {


/**
 * A variant type designed to store checkpoint metadata values.
 *
 * This class provides a type-safe way to hold checkpoint metadata values.
 * It supports multiple data types such as "booleam" (bool), "integer" (Long),
 * "unsigned" (ULong), "double", and "string". The class also allows specifying
 * the intended storage type for use in file formats like GGUF.
 *
 * Explicit constructors are provided for each supported type to ensure clear
 * and intuitive initialization. Additionally, conversion methods allow
 * retrieving the stored value in different types.
 * 
 * 
 * Example:
 * @code
 * #include <tin/metavariant.h>
 * using namespace tin;
 *
 * // Create metadata values using explicit constructors
 * MetaVariant flag{ true };                       // Boolean
 * MetaVariant count{ 42L };                       // Integer
 * MetaVariant pi{ 3.14159, StorageType::STRING }; // Double stored as string
 * MetaVariant message{ "hello" };                 // String
 *
 * // Retrieve values using conversion methods
 * auto flagAsInt     = flag.as_integer();   // Returns 1L
 * auto countAsText   = count.as_string();   // Returns "42"
 * auto piAsUnsigned  = pi.as_unsigned();    // Returns 3UL
 * auto text          = message.as_string(); // Returns "hello"
 *
 * // Check how the value was originally stored
 * StorageType st1 = flag.storage_type();    // Returns StorageType::BOOL
 * StorageType st2 = pi.storage_type();      // Returns StorageType::STRING
 * @endcode
 * 
 */
class MetaVariant
{
// CONSTANT VALUES
public:
    static const MetaVariant None;   

// CONSTRUCTION
public:
    explicit MetaVariant(bool       value , StorageType storageType = StorageType::BOOL   ) noexcept;
    explicit MetaVariant(Long       value , StorageType storageType = StorageType::INT32  ) noexcept;
    explicit MetaVariant(ULong      value , StorageType storageType = StorageType::UINT32 ) noexcept;
    explicit MetaVariant(double     value , StorageType storageType = StorageType::FLOAT32) noexcept;
    explicit MetaVariant(StringView value , StorageType storageType = StorageType::STRING ) noexcept;
    //explicit MetaVariant(Vector<String> vector, StorageType storageType = StorageType::ARRAY_OF_STRINGS ) noexcept;
    explicit MetaVariant(MetaVariant&& other) noexcept = default;
    explicit MetaVariant(const MetaVariant& other) = default;
private:
    MetaVariant() noexcept;


// CONVERSIONS
public:
    bool   as_boolean (bool       default_ = false) const noexcept;
    Long   as_integer (Long       default_ = 0    ) const noexcept;
    ULong  as_unsigned(ULong      default_ = 0    ) const noexcept;
    double as_double  (double     default_ = 0.0  ) const noexcept;
    String as_string  (StringView default_ = ""   ) const noexcept;
    //Vector<String> as_string_vector() const noexcept;
    
// TYPE HINTS
public:
    StorageType storage_type() const {
        return _storageType;
    }

// IMPLEMENTATION
private:
    using      Variant = std::variant<bool, Long, ULong, double, String>;
    enum class Type { NONE, BOOL, LONG, ULONG, DOUBLE, STRING
                      /*STRING_VECTOR, METAVARIANT_VECTOR*/ };
private:
    Type        _type;
    Variant     _value;
    StorageType _storageType;
};


}      // namespace tin
#endif // TIN_METAVARIANT_H_
