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
#include <fstream>     // for std::ifstream
#include <algorithm>   // for std::sort
#include <tin/tensormap.h>
namespace tin {

//======================= CONSTRUCTION/DESTRUCTION ========================//

/**
 * Construct a TensorMap object from an rvalue vector of tensors and metadata.
 *
 * This constructor initializes a `TensorMap` object using a collection of 
 * `TensorInfo` objects provided in the input vector, along with associated 
 * `Metadata`. It moves ownership of the tensors and metadata for efficient 
 * resource management. Additionally, it allows specifying an optional position 
 * offset to adjust the raw data boundaries of each tensor.
 *
 * @param tensors       An rvalue reference to a vector containing `TensorInfo`
 *                      objects representing individual tensors.
 * @param metadata      An rvalue reference to a `Metadata` object holding information
 *                      loaded from the checkpoint file associated with the tensors.
 * @param rawDataOffset An optional stream position offset applied to both 
 *                      `rawDataBegin` and `rawDataEnd` properties of each tensor. 
 *                      The default value is 0 if not specified.
 */
TensorMap::TensorMap(std::vector<TensorInfo>&& tensors,
                     Metadata&&                metadata,
                     std::streampos            rawDataOffset // = 0
                     
) noexcept
: _metadata{ std::move(metadata) }
{
    _map.reserve( tensors.size() );
    for( auto& tensor : tensors ) {
        TensorInfo adjustedTensorInfo{ std::move(tensor), rawDataOffset };
        _map.emplace( adjustedTensorInfo.name(), adjustedTensorInfo );
    }
}

/**
 * Constructs a TensorMap object from a vector of tensors and checkpoint metadata.
 *
 * This constructor initializes a `TensorMap` object with the collection of
 * tensors provided in the input vector, along with associated metadata from
 * the checkpoint file. It also allows specifying a position offset to adjust
 * the raw data boundaries of each tensor.
 * 
 * @param tensors       A vector containing `TensorInfo` objects representing
 *                      the individual tensors.
 * @param metadata      A `Metadata` object holding information loaded from 
 *                      the checkpoint file associated with the tensors.
 * @param rawDataOffset An optional stream position offset applied to both 
 *                      `rawDataBegin` and `rawDataEnd` properties of each tensor.
 *                      If not specified, the default value is 0.
 */
TensorMap::TensorMap(const std::vector<TensorInfo>& tensors,
                     const Metadata&                metadata,
                     std::streampos                 rawDataOffset // = 0
) 
: _metadata{ metadata }
{
    _map.reserve( tensors.size() );
    for( const auto& tensor : tensors ) {
        _map.emplace( tensor.name(), TensorInfo{ tensor, rawDataOffset } );
    }
}

//=========================== LOADING TENSORMAP ===========================//

/**
 * Creates a TensorMap from a file.
 *
 * This function attempts to read and create a `TensorMap` object from the
 * file located at `filePath`. If an error occurs during this process, the
 * `outError` parameter is set accordingly, and an empty `TensorMap` is
 * returned.
 *
 * @param      filePath The path to the .safetensors or .gguf file.
 * @param[out] outError The variable that receives the error code.
 *                      (It will be set to `ReadError::None` if no error occurs)
 * @return
 *     A populated TensorMap object if the file is successfully read and parsed,
 *     or an empty TensorMap in case of errors.
 */
TensorMap
TensorMap::from_file(const Path& filePath,
                     ReadError&  outError
) noexcept {
    if( !std::filesystem::exists(filePath) ) {
        outError = ReadError::FileNotFound; return TensorMap{};
    }
    auto istream  = std::ifstream{ filePath, std::ios_base::binary };
    auto fileSize = static_cast<std::streampos>( std::filesystem::file_size(filePath) );
    return TensorMap::from_stream(istream, outError, filePath, fileSize);
}

/**
 * Creates a TensorMap from an input stream.
 *
 * This function attempts to create a `TensorMap` object by reading data from
 * the provided input stream `istream`. It determines whether the file is in
 * GGUF or safetensors format. If an error occurs during this process, the
 * `outError` parameter is set accordingly.
 *
 * @param      istream  The input stream from which data will be read.
 * @param[out] outError The variable that receives the error code.
 *                      (It will be set to `ReadError::None` if no error occurs)
 * @param      filePath The full path of the original file used to create the input stream.
 *                      (If the `istream` does not originate from a file, this parameter should be empty)
 * @param      fileSize The total size of the file associated with the input stream.
 *                      (Can be 0 but no validation of offsets or sizes will be performed) 
 * @return
 *     A populated TensorMap object if the file is successfully read and parsed,
 *     or an empty TensorMap in case of errors.
 */
TensorMap
TensorMap::from_stream(std::istream&   istream,
                       ReadError&      outError,
                       const Path&     filePath,  // = {},
                       std::streamsize fileSize   // = 0
) noexcept {

    char     firstChars[8];
    uint8_t* firstBytes = reinterpret_cast<uint8_t*>(firstChars);

    // read the first 8 bytes of istream
    istream.read(firstChars, 8);
    const auto ggufIdentifier = std::string_view(firstChars, 4);
    const auto ggufVersion    = firstBytes[4] + firstBytes[5] * 256 + firstBytes[6] * (256*256) + firstBytes[7] * (256*256*256);


    outError = ReadError::None;
    if( ggufIdentifier == "GGUF" && 1 <= ggufVersion && ggufVersion <= 5)  {
        return _fromgguf(firstBytes, istream, outError, filePath, fileSize);
    } else {
        return _fromsafetensors(firstBytes, istream, outError, filePath, fileSize);
    }
}

//=========================== QUERYING THE MAP ============================//

const TensorInfo&
TensorMap::operator[](const String& name) const noexcept {
    auto it = _map.find(name);
    return it == end() ? TensorInfo::None : it->second;
}

TensorMap::const_iterator
TensorMap::find(const String& name) const noexcept {
    return _map.find(name);
}

bool
TensorMap::contains(const String& name) const noexcept {
    return _map.contains(name);
}

/**
 * Collects tensor information from the map, with optional sorting.
 *
 * This function returns a vector containing all `TensorInfo` objects from
 * the map. The collection can optionally be sorted based on a provided
 * criteria.
 *
 * @param sortBy An optional value that determines the sorting criteria. 
 *    Possible values include:
 *      - `SortBy::NAME`          : Sorts by tensor name in ascending order.
 *      - `SortBy::NAME_AND_INDEX': Sorts by tensor name but considers indices.
 *      - `SortBy::DTYPE`         : Sorts by data type in ascending order.
 *      - `SortBy::BYTES`         : Sorts by raw data size in bytes in ascending order.
 *      - `SortBy::NUMEL`         : Sorts by the number of elements in ascending order.
 *      - `SortBy::UNSORTED`      : No sorting is applied, very fast.
 *
 * @return A vector containing copies of each `TensorInfo` object from the map.
 *         The elements in the vector will be sorted according to the specified
 *         criteria or in an undefined order if no sorting is requested.
 */
std::vector<TensorInfo>
TensorMap::collect_tensors(SortBy sortBy // = SortBy::UNSORTED
) const noexcept
{
    class Wrapper {
        TensorInfo       _tensorInfo;
        ULong            _cachedNumel;
        std::streamsize  _cachedBytes;
        mutable String   _cachedNormalName;
    public:
        Wrapper(const TensorInfo& tensorInfo)
        : _tensorInfo{tensorInfo}, _cachedNumel{ tensorInfo.numel() }, _cachedBytes{ tensorInfo.raw_data_size() }
        {}
        const TensorInfo& unwrap() const noexcept { return _tensorInfo;         }
        StringView        name()   const noexcept { return _tensorInfo.name();  }
        DType             dtype()  const noexcept { return _tensorInfo.dtype(); }
        std::streamsize   bytes()  const noexcept { return _cachedBytes;        }
        ULong             numel()  const noexcept { return _cachedNumel;        }
        const String& normalName() const noexcept {
            if( _cachedNormalName.empty() ) { _cachedNormalName = _tensorInfo.generate_normalized_name(); }
            return _cachedNormalName;
        }
    };
    using WrapperPtr = std::unique_ptr<Wrapper>;


    std::vector<TensorInfo> result;
    result.reserve( _map.size() );

    if( sortBy == SortBy::UNSORTED ) {
        // NO SORTING:
        // just copy all TensorInfo objects from the map to the result vector
        for( const auto& pair : _map ) {
            result.emplace_back( pair.second );
        }
    }
    else {
        // SORTING:
        // create a vector of pointers to Wrapper objects,
        // (each wrapper contains a TensorInfo with additional cached values)
        std::vector< WrapperPtr > wrapperPointers;
        wrapperPointers.reserve( _map.size() );
        for( const auto& pair : _map ) {
            wrapperPointers.emplace_back( new Wrapper( pair.second ) );
        }
        // sort the vector of pointers based on the specified sorting criteria
        std::sort( wrapperPointers.begin(), wrapperPointers.end(),
                    [sortBy](const WrapperPtr& a, const WrapperPtr& b)
            {
                switch( sortBy ) {
                    case SortBy::NAME:           return a->name()       <  b->name();
                    case SortBy::NAME_AND_INDEX: return a->normalName() <  b->normalName();
                    case SortBy::DTYPE:          return a->dtype()      <  b->dtype();
                    case SortBy::BYTES:          return a->bytes()      <  b->bytes();
                    case SortBy::NUMEL:          return a->numel()      <  b->numel();
                    default:
                        return false;
                }
            }
        );
        // at this point `wrapperPointers` is sorted
        // so we can just copy them to the result vector of TensorInfo objects
        for( const auto& wrapperPtr : wrapperPointers ) {
            result.emplace_back( std::move( wrapperPtr->unwrap() ) );
        }
    }
    return result;
}


} // namespace tin