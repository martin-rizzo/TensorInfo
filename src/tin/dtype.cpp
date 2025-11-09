/*
| File    : dtype.cpp
| Purpose : Enumerate different data types used for tensor storage.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 7, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <unordered_map>
#include <tin/dtype.h>
namespace tin {


} // namespace tin


std::string_view
to_string(tin::DType type) {
    static const std::unordered_map<tin::DType, std::string_view> typeMap = {
        {tin::DType::F64    , "F64"    },
        {tin::DType::F32    , "F32"    },
        {tin::DType::F16    , "F16"    },
        {tin::DType::BF16   , "BF16"   },
        {tin::DType::F8_E8M0, "F8_E8M0"},
        {tin::DType::F8_E5M2, "F8_E5M2"},
        {tin::DType::F8_E4M3, "F8_E4M3"},
        {tin::DType::F6_E3M2, "F6_E3M2"},
        {tin::DType::F6_E2M3, "F6_E2M3"},
        {tin::DType::F4     , "F4"     },
        {tin::DType::I64    , "I64"    },
        {tin::DType::I32    , "I32"    },
        {tin::DType::I16    , "I16"    },
        {tin::DType::I8     , "I8"     },
        {tin::DType::U64    , "U64"    },
        {tin::DType::U32    , "U32"    },
        {tin::DType::U16    , "U16"    },
        {tin::DType::U8     , "U8"     },
        {tin::DType::BOOL   , "BOOL"   },
        {tin::DType::Q6_K   , "Q6_K"   },
        {tin::DType::Q5_K   , "Q5_K"   },
        {tin::DType::Q4_K   , "Q4_K"   },
        {tin::DType::Q3_K   , "Q3_K"   },
        {tin::DType::Q2_K   , "Q2_K"   },
        {tin::DType::IQ4_NL , "IQ4_NL" },
        {tin::DType::IQ4_XS , "IQ4_XS" },
        {tin::DType::IQ3_S  , "IQ3_S"  },
        {tin::DType::IQ3_XXS, "IQ3_XXS"},
        {tin::DType::IQ2_S  , "IQ2_S"  },
        {tin::DType::IQ2_XS , "IQ2_XS" },
        {tin::DType::IQ2_XXS, "IQ2_XXS"},
        {tin::DType::IQ1_M  , "IQ1_M"  },
        {tin::DType::IQ1_S  , "IQ1_S"  },
        {tin::DType::Q8_1   , "Q8_1"   },
        {tin::DType::Q8_0   , "Q8_0"   },
        {tin::DType::Q5_1   , "Q5_1"   },
        {tin::DType::Q5_0   , "Q5_0"   },
        {tin::DType::Q4_1   , "Q4_1"   },
        {tin::DType::Q4_0   , "Q4_0"   }
    };
    // find the string representation in the map
    auto it = typeMap.find(type);
    return it != typeMap.end() ? it->second : "Unknown";
}
