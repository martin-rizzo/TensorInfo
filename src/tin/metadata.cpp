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
#include <tin/metadata.h>
namespace tin {

//======================= MODIFYING METADATA VALUES =======================//

void
Metadata::set_boolean(StringView  key,
                      bool        value,
                      StorageType storageType // = StorageType::BOOL
){
    _map.emplace(key, MetaVariant(value, storageType));
}

void
Metadata::set_integer(StringView  key,
                      long int    value,
                      StorageType storageType // = StorageType::INT32
){
    _map.emplace(key, MetaVariant(value, storageType));
}

void
Metadata::set_unsigned(StringView    key,
                       long unsigned value,
                       StorageType   storageType // = StorageType::UINT32
){
    _map.emplace(key, MetaVariant(value, storageType));
}

void
Metadata::set_float(StringView  key,
                    double      value,
                    StorageType storageType  // = StorageType::FLOAT32
){
    _map.emplace(key, MetaVariant(value, storageType));
}

void
Metadata::set_string(StringView  key,
                     StringView  value,
                     StorageType storageType  // = StorageType::STRING
){
    _map.emplace(key, MetaVariant(value, storageType));
}

//======================= ACCESSING METADATA VALUES =======================//

const MetaVariant&
Metadata::get(StringView key
) const noexcept {
    auto it = _map.find( String{key} );
    return( it == _map.end() ) ? MetaVariant::None : it->second;
}

bool
Metadata::contains(StringView key
) const noexcept {
    return _map.contains( String{key} );
}


} // namespace tin