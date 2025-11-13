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
#include <tin/metavalue.h>
namespace tin {


class Metadata
{
public:
    // Type aliases for compatibility with standard containers
    using key_type        = String;
    using mapped_type     = MetaValue;
    using value_type      = std::pair<const key_type, mapped_type>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    // Iterators
    using InternalMap    = std::unordered_map<key_type, mapped_type>;
    using iterator       = InternalMap::iterator;
    using const_iterator = InternalMap::const_iterator;

// MODIFIER METHODS
public:
    void set_boolean (StringView key, bool          value);
    void set_integer (StringView key, long int      value, MetaType metatype = MetaType::INT32  );
    void set_unsigned(StringView key, long unsigned value, MetaType metatype = MetaType::UINT32 );
    void set_float   (StringView key, double        value, MetaType metatype = MetaType::FLOAT32);
    void set_string  (StringView key, StringView    value);

// LOOKUP METHODS
public:
    [[nodiscard]] const MetaValue& get(StringView key) const noexcept;
    [[nodiscard]] bool             contains(StringView key) const noexcept;


// ??
public:
    std::vector<const key_type*> keys() const;


public:
    MetaType get_original_type(StringView key) const;
    MetaType get_preferred_type(StringView key) const;

public:
    

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
