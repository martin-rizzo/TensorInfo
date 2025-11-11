/*
| File    : tensormap.h
| Purpose : Maps tensor names to their metadata (TensorInfo)
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_TENSORMAP_H_
#define TIN_TENSORMAP_H_
#include <istream>          // for std::istream
#include <unordered_map>    // for std::unordered_map
#include <tin/common.h>     // for String, Path
#include <tin/tensorinfo.h> // for TensorInfo
namespace tin {


/**
 * Maps tensor names to their metadata (TensorInfo)
 * 
 * This class provides map-like functionality for managing tensor metadata,
 * supporting standard map operations including insertion, lookup, removal,
 * and iteration.
 */
class TensorMap
{
public:
    // Type aliases for compatibility with standard containers
    using key_type        = String;
    using mapped_type     = TensorInfo;
    using value_type      = std::pair<const key_type, mapped_type>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    // Iterators
    using iterator       = std::unordered_map<String, TensorInfo>::iterator;
    using const_iterator = std::unordered_map<String, TensorInfo>::const_iterator;


// CONSTRUCTION/DESTRUCTION
public:
    TensorMap() = default;
    ~TensorMap() = default;

// COPY/MOVE
public:
    TensorMap(const TensorMap& other) = default;
    TensorMap(TensorMap&& other) noexcept = default;
    TensorMap& operator=(const TensorMap& other) = default;
    TensorMap& operator=(TensorMap&& other) noexcept = default;

// LOADING
public:
    static TensorMap from_file(const Path& filePath);
    static TensorMap from_stream(std::istream& istream, const Path& filePath = {}, std::streamsize fileSize = 0);

// MODIFIERS
public:
    std::pair<iterator, bool> insert( const TensorInfo& tensorInfo ) {
        return _map.emplace(tensorInfo.name(), tensorInfo);
    }
    

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
    static TensorMap _fromsafetensors(const uint8_t firstBytes[8], std::istream& istream, const Path& filePath = {}, std::streamsize fileSize = 0, std::streampos byteBufferPosition = 0);
    static TensorMap _fromgguf(const uint8_t firstBytes[8], std::istream& istream, const Path& filePath = {}, std::streamsize fileSize = 0, std::streampos byteBufferPosition = 0);
private:
    std::unordered_map<String, TensorInfo> _map;
};


}      // namespace tin
#endif // TIN_TENSORMAP_H_
