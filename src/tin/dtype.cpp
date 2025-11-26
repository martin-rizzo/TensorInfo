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

DType DType::UNKNOWN(DType::Value::UNKNOWN);
DType DType::F64    (DType::Value::F64    );
DType DType::F32    (DType::Value::F32    );
DType DType::F16    (DType::Value::F16    );
DType DType::BF16   (DType::Value::BF16   );
DType DType::I64    (DType::Value::I64    );
DType DType::I32    (DType::Value::I32    );
DType DType::I16    (DType::Value::I16    );
DType DType::I8     (DType::Value::I8     );
DType DType::U64    (DType::Value::U64    );
DType DType::U32    (DType::Value::U32    );
DType DType::U16    (DType::Value::U16    );
DType DType::U8     (DType::Value::U8     );
DType DType::BOOL   (DType::Value::BOOL   );
DType DType::F8_E8M0(DType::Value::F8_E8M0);
DType DType::F8_E5M2(DType::Value::F8_E5M2);  
DType DType::F8_E4M3(DType::Value::F8_E4M3);
DType DType::F6_E3M2(DType::Value::F6_E3M2);
DType DType::F6_E2M3(DType::Value::F6_E2M3);
DType DType::F4     (DType::Value::F4     );
DType DType::Q6_K   (DType::Value::Q6_K   );  
DType DType::Q5_K   (DType::Value::Q5_K   );
DType DType::Q4_K   (DType::Value::Q4_K   );
DType DType::Q3_K   (DType::Value::Q3_K   );
DType DType::Q2_K   (DType::Value::Q2_K   );
DType DType::IQ4_NL (DType::Value::IQ4_NL );
DType DType::IQ4_XS (DType::Value::IQ4_XS );  
DType DType::IQ3_S  (DType::Value::IQ3_S  );
DType DType::IQ3_XXS(DType::Value::IQ3_XXS);
DType DType::IQ2_S  (DType::Value::IQ2_S  );
DType DType::IQ2_XS (DType::Value::IQ2_XS );  
DType DType::IQ2_XXS(DType::Value::IQ2_XXS);
DType DType::IQ1_M  (DType::Value::IQ1_M  );
DType DType::IQ1_S  (DType::Value::IQ1_S  );  
DType DType::Q8_1   (DType::Value::Q8_1   );
DType DType::Q8_0   (DType::Value::Q8_0   );  
DType DType::Q5_1   (DType::Value::Q5_1   );
DType DType::Q5_0   (DType::Value::Q5_0   );
DType DType::Q4_1   (DType::Value::Q4_1   );
DType DType::Q4_0   (DType::Value::Q4_0   );

/**
 * Converts a `DType` enumeration value to its corresponding string representation.
 *
 * This function provides a standardized method for converting `DType`
 * enumeration values into their respective string representations. It is
 * defined within the `std` namespace, offering an overload of `std::to_string`
 * that is specifically tailored for `DType` values. 
 *
 * @param type The `DType` enumeration value to be converted to a string.
 * @return A string view representing the name of the data type. Returns "Unknown"
 *         if the provided `type` does not have an associated string representation in
 *         the predefined mapping.
 */
std::string_view
DType::to_string() const noexcept
{
    static const std::unordered_map<DType::Value, std::string_view> typeMap = {
        {DType::Value::F64    , "F64"    },
        {DType::Value::F32    , "F32"    },
        {DType::Value::F16    , "F16"    },
        {DType::Value::BF16   , "BF16"   },
        {DType::Value::F8_E8M0, "F8_E8M0"},
        {DType::Value::F8_E5M2, "F8_E5M2"},
        {DType::Value::F8_E4M3, "F8_E4M3"},
        {DType::Value::F6_E3M2, "F6_E3M2"},
        {DType::Value::F6_E2M3, "F6_E2M3"},
        {DType::Value::F4     , "F4"     },
        {DType::Value::I64    , "I64"    },
        {DType::Value::I32    , "I32"    },
        {DType::Value::I16    , "I16"    },
        {DType::Value::I8     , "I8"     },
        {DType::Value::U64    , "U64"    },
        {DType::Value::U32    , "U32"    },
        {DType::Value::U16    , "U16"    },
        {DType::Value::U8     , "U8"     },
        {DType::Value::BOOL   , "BOOL"   },
        {DType::Value::Q6_K   , "Q6_K"   },
        {DType::Value::Q5_K   , "Q5_K"   },
        {DType::Value::Q4_K   , "Q4_K"   },
        {DType::Value::Q3_K   , "Q3_K"   },
        {DType::Value::Q2_K   , "Q2_K"   },
        {DType::Value::IQ4_NL , "IQ4_NL" },
        {DType::Value::IQ4_XS , "IQ4_XS" },
        {DType::Value::IQ3_S  , "IQ3_S"  },
        {DType::Value::IQ3_XXS, "IQ3_XXS"},
        {DType::Value::IQ2_S  , "IQ2_S"  },
        {DType::Value::IQ2_XS , "IQ2_XS" },
        {DType::Value::IQ2_XXS, "IQ2_XXS"},
        {DType::Value::IQ1_M  , "IQ1_M"  },
        {DType::Value::IQ1_S  , "IQ1_S"  },
        {DType::Value::Q8_1   , "Q8_1"   },
        {DType::Value::Q8_0   , "Q8_0"   },
        {DType::Value::Q5_1   , "Q5_1"   },
        {DType::Value::Q5_0   , "Q5_0"   },
        {DType::Value::Q4_1   , "Q4_1"   },
        {DType::Value::Q4_0   , "Q4_0"   }
    };
    // find the string representation in the map
    auto it = typeMap.find(_value);
    return it != typeMap.end() ? it->second : "Unknown";
}

} // namespace tin
