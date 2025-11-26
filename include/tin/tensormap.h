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
#include <tin/sortby.h>     // for SortBy
#include <tin/readerror.h>  // for ReadError
#include <tin/tensorinfo.h> // for TensorInfo
#include <tin/metadata.h>   // for Metadata
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
 // Type aliases for compatibility with standard containers
public:
    using key_type        = String;
    using mapped_type     = TensorInfo;
    using value_type      = std::pair<const key_type, mapped_type>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using iterator        = std::unordered_map<String, TensorInfo>::iterator;
    using const_iterator  = std::unordered_map<String, TensorInfo>::const_iterator;


// CONSTRUCTION/DESTRUCTION
public:
    TensorMap(std::vector<TensorInfo>&& tensors, Metadata&& metadata, std::streampos tensorRawDataOffset = 0) noexcept;
    TensorMap(const std::vector<TensorInfo>& tensors, const Metadata& metadata, std::streampos tensorRawDataOffset = 0);
    TensorMap() noexcept = default;
    ~TensorMap() noexcept = default;

// COPY/MOVE
public:
    TensorMap(const TensorMap& other) = default;
    TensorMap(TensorMap&& other) noexcept = default;
    TensorMap& operator=(const TensorMap& other) = default;
    TensorMap& operator=(TensorMap&& other) noexcept = default;

// LOADING TENSORMAP
public:
    [[nodiscard]] static TensorMap from_file(const Path& filePath, ReadError& outError) noexcept;
    [[nodiscard]] static TensorMap from_stream(std::istream& istream, ReadError& outError, const Path& filePath = {}, std::streamsize fileSize = 0) noexcept;


// QUERYING THE MAP
public:
    [[nodiscard]] const TensorInfo& operator[](const String& name) const noexcept;
    [[nodiscard]] const_iterator find(const String& name) const noexcept;
    [[nodiscard]] bool           contains(const String& name) const noexcept;
    [[nodiscard]] std::vector<TensorInfo> collect_tensors(SortBy sortBy = SortBy::NONE) const noexcept;


// MODIFYING THE MAP
public:
  
    void clear() noexcept {
        _map.clear();
    }

    std::pair<iterator, bool> insert( const TensorInfo& tensorInfo ) {
        return _map.emplace(tensorInfo.name(), tensorInfo);
    }

// METADATA ACCESS
public:
    [[nodiscard]] Metadata&       metadata() noexcept;
    [[nodiscard]] const Metadata& metadata() const noexcept;
    
// ITERATORS
public:
    [[nodiscard]] iterator       begin()  noexcept;
    [[nodiscard]] const_iterator begin()  const noexcept;
    [[nodiscard]] const_iterator cbegin() const noexcept;
    [[nodiscard]] iterator       end()    noexcept;
    [[nodiscard]] const_iterator end()    const noexcept;
    [[nodiscard]] const_iterator cend()   const noexcept;

// IMPLEMENTATION
private:
    static TensorMap _fromsafetensors(const uint8_t firstBytes[8], std::istream& istream, ReadError& outError, const Path& filePath = {}, std::streamsize fileSize = 0, std::streampos byteBufferPosition = 0) noexcept;
    static TensorMap _fromgguf(const uint8_t firstBytes[8], std::istream& istream, ReadError& outError, const Path& filePath = {}, std::streamsize fileSize = 0, std::streampos byteBufferPosition = 0) noexcept;
private:
    std::unordered_map<String, TensorInfo> _map;
    Metadata                               _metadata;
};


//======================= INLINES: METADATA ACCESS ========================//

inline Metadata&
TensorMap::metadata() noexcept { return _metadata; }

inline const Metadata&
TensorMap::metadata() const noexcept { return _metadata; }


//========================== INLINES: ITERATORS ===========================//

/** Returns an iterator to the beginning of the map. */
inline TensorMap::iterator
TensorMap::begin() noexcept { return _map.begin();  }

/** Returns an iterator to the beginning of the const map. */
inline TensorMap::const_iterator
TensorMap::begin() const noexcept { return _map.begin();  }

/** Returns an explicit constant iterator to the beginning of the map. */
inline TensorMap::const_iterator
TensorMap::cbegin() const noexcept { return _map.cbegin(); }

/** Returns an iterator to the end of the map. */
inline TensorMap::iterator
TensorMap::end() noexcept { return _map.end(); }

/** Returns an iterator to the end of the const map. */
inline TensorMap::const_iterator
TensorMap::end() const noexcept { return _map.end(); }

/** Returns an explicit constant iterator to the end of the map. */
inline TensorMap::const_iterator
TensorMap::cend() const noexcept { return _map.cend(); }



}      // namespace tin
#endif // TIN_TENSORMAP_H_
