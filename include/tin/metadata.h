/*
| File    : metadata.h
| Purpose : Store additional information about the checkpoint.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 12, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_METADATA_H_
#define TIN_METADATA_H_
#include <unordered_map>
#include <tin/common.h>
#include <tin/metavariant.h>
namespace tin {


class Metadata
{
// Type aliases for compatibility with standard containers
public:
    using key_type        = String;
    using mapped_type     = MetaVariant;
    using value_type      = std::pair<const key_type, mapped_type>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using InternalMap     = std::unordered_map<key_type, mapped_type>;
    using iterator        = InternalMap::iterator;
    using const_iterator  = InternalMap::const_iterator;

// CONSTRUCTION/DESTRUCTION
public:
    Metadata() noexcept  = default;
    ~Metadata() noexcept = default;
    Metadata(Metadata&& other) noexcept = default;
    Metadata(const Metadata& other) = default;


// MODIFYING METADATA VALUES
public:
    void set_boolean (StringView key, bool       value, StorageType storageType = StorageType::BOOL   );
    void set_integer (StringView key, Long       value, StorageType storageType = StorageType::INT32  );
    void set_unsigned(StringView key, ULong      value, StorageType storageType = StorageType::UINT32 );
    void set_float   (StringView key, double     value, StorageType storageType = StorageType::FLOAT32);
    void set_string  (StringView key, StringView value, StorageType storageType = StorageType::STRING );

// ACCESSING METADATA VALUES
public:
    [[nodiscard]] const MetaVariant& get(StringView key) const noexcept;
    [[nodiscard]] bool contains(StringView key) const noexcept;


// ??
public:
    std::vector<const key_type*> keys() const;

   
// ITERATORS
public:

    /**
     * Returns an iterator to the beginning of the map.
     */
    [[nodiscard]] iterator begin() noexcept { return _map.begin();  }

    /**
     * Returns an iterator to the beginning of the const map.
     */
    [[nodiscard]] const_iterator begin() const noexcept { return _map.begin();  }

    /**
     * Returns an explicit constant iterator to the beginning of the map.
     */
    [[nodiscard]] const_iterator cbegin() const noexcept { return _map.cbegin(); }

    /**
     * Returns an iterator to the end of the map.
     */
    [[nodiscard]] iterator end() noexcept { return _map.end(); }

    /**
     * Returns an iterator to the end of the const map.
     */
    [[nodiscard]] const_iterator end() const noexcept { return _map.end(); }

    /**
     * Returns an explicit constant iterator to the end of the map.
     */
    [[nodiscard]] const_iterator cend() const noexcept { return _map.cend(); }


// IMPLEMENTATION
private:
    InternalMap _map;
};


}      // namespace tin
#endif // TIN_METADATA_H_
