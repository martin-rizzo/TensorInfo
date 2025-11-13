<div align="center">
  
# 🧭 TensorInfo (tin::)
[![Platform](https://img.shields.io/badge/language%3A-C++20-B8860B)](#)
[![Platform](https://img.shields.io/badge/platform%3A-multiplatform-007BFF)](#)
[![License](https://img.shields.io/github/license/martin-rizzo/TensorInfo?label=license%3A&color=28A745)](#)
[![Version](https://img.shields.io/github/v/tag/martin-rizzo/TensorInfo?label=version%3A&color=D07250)](#)
[![Last](https://img.shields.io/github/last-commit/martin-rizzo/TensorInfo?label=last%20commit%3A)](#)  
<img src="phase0.png" width="50%"></img>

</div>

> [!NOTE]
> This project is in "Phase 0". At this point, I'm exploring various approaches for implementation without a predetermined path. Consequently, features are still flexible and subject to change as the project develops further.

**TensorInfo** is a lightweight C++ library designed to simplify the handling of tensor data stored in specialized file formats such as `.safetensors` and `.gguf`. It provides access to tensor metadata, shapes, and memory offsets, enabling seamless integration with machine learning checkpoints verifing and data processing workflows.

It includes utilities for inspecting safetensors/gguf files:
- `lstensors`: CLI tool to list tensors and metadata in safetensors and gguf files.

## Key Features

- Load and parse tensor information from `.safetensors` and `.gguf` files.
- Access tensor metadata, including shapes, types, and byte offsets.
- Access checkpoint metadata, including model configurations and additional information.
- Simple design for easy integration with existing projects.
- Thoroughly documented code at a tutorial-like level to facilitate learning.
- Crafted with all the love and unrelenting suffering that only C++20 can offer.
- CLI utilities

## Overview

TensorInfo allows developers to load and inspect tensor maps from different file formats. The library currently includes two core classes:
- **TensorMap**: A high-level interface for managing tensor data and metadata.
- **TensorTree**: A hierarchical navigation system for exploring tensor structures within the loaded files.

## Managing Checkpoint Metadata
To access checkpoint metadata, utilize __`tensorMap.metadata()`__ as it provides the necessary methods.

### Loading Metadata  
To load metadata, follow the same procedure as for loading regular checkpoint data:
```cpp
#include <tin/tensormap.h>
using namespace tin;

void my_function(void) {
    ReadError readError;

    // Attempt to load the file
    const auto tensorMap = TensorMap::from_file("/path/to/checkpoint.gguf", readError);
    if( readError != ReadError::None ) { 
        fatal_read_error(readError);
    }
    ....
}
```

### Accessing Metadata Values
Use the __`get`__ method in conjunction with an appropriate __`as_*()`__ method for the desired data type:
```cpp
// Retrieve "llama.context_length" as an integer if you know this key exists
auto contextLength = tensorMap.metadata().get("llama.context_length").as_integer();

// Each of the `as_*()` methods can have a default value used when the key does not exist
auto exampleBool     = tensorMap.metadata().get("example.bool").as_boolean(true);
auto exampleInteger  = tensorMap.metadata().get("example.integer").as_integer(-200);
auto exampleUnsigned = tensorMap.metadata().get("example.unsigned").as_unsigned(0);
auto exampleFloat    = tensorMap.metadata().get("example.float").as_float(12.3);
auto exampleString   = tensorMap.metadata().get("example.string").as_string("empty");
```

### Modifying Metadata Values  
To update or set metadata values, use methods such as `set_integer`, specifying the key and the new value.

**Example:**  
```cpp
// Set `"llama.context_length"` to `8192`.
tensorMap.metadata().set_integer("llama.context_length", 8 * 1024);
```

### Enumerating Metadata Values  
To iterate through all metadata entries, use a loop to access each as a key-value pair.

**Example:**  
```cpp
// Print each metadata key and its string representation.
std::cout << "Metadata:" << std::endl;
for (auto& it : tensorMap.metadata()) {
    std::cout << "  " << it.first << ": " << it.second.to_string() << std::endl;
}
```

## License

Copyright (c) 2025 Martin Rizzo\
This project is licensed under the MIT license.\
See the ["LICENSE"](LICENSE) file for details.
