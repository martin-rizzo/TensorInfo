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
Metadata::set_boolean(StringView key,
                      bool      value
){
    _map.emplace(key, MetaValue(value));
}

void
Metadata::set_integer(StringView key,
                      long int   value,
                      MetaType   metatype // = MetaType::INT32
){
    _map.emplace(key, MetaValue(value, metatype));
}

void
Metadata::set_unsigned(StringView    key,
                       long unsigned value,
                       MetaType      metatype // = MetaType::UINT32
){
    _map.emplace(key, MetaValue(value, metatype));
}

void
Metadata::set_float(StringView key,
                    double     value,
                    MetaType   metatype  // = MetaType::FLOAT32
){
    _map.emplace(key, MetaValue(value, metatype));
}

void
Metadata::set_string(StringView key,
                     StringView value
){
    _map.emplace(key, MetaValue(value));
}

//======================= ACCESSING METADATA VALUES =======================//

const MetaValue&
Metadata::get(StringView key
) const noexcept {
    auto it = _map.find( String{key} );
    return( it == _map.end() ) ? MetaValue::None : it->second;
}

bool
Metadata::contains(StringView key
) const noexcept {
    return _map.contains( String{key} );
}


} // namespace tin