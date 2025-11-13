<div align="center">
  
# 🧭 TensorInfo (tin::)
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
- Implemented in C++20 for the delight of a select few and as a nightmare for the rest of us.
- CLI utilities

## Overview

TensorInfo allows developers to load and inspect tensor maps from different file formats. The library currently includes two core classes:
- **TensorMap**: A high-level interface for managing tensor data and metadata.
- **TensorTree**: A hierarchical navigation system for exploring tensor structures within the loaded files.

## Managing Checkpoint Metadata

### Loading Metadata  
To load metadata, follow the same procedure as for loading regular checkpoint data.

**Example:**
```cpp
ReadError readError;

// Load the checkpoint file
const auto tensorMap = TensorMap::from_file("/path/to/checkpoint.safetensors", readError);
if (readError != ReadError::None) { fatal_read_error(readError); }
```

### Accessing Metadata Values  
To access checkpoint metadata, utilize `tensorMap.metadata()`, which provides the necessary methods.\
To retrieve a specific value, use the `get` method by specifying the key name and providing a default if needed.

**Example:**  
```cpp
// Retrieve an integer for `"llama.context_length"`, using `defvalue` if absent.
auto contextLength = tensorMap.metadata().get("llama.context_length").as_integer(defvalue);

// Additional examples using as_string and as_float
auto exampleString = tensorMap.metadata().get("example.key").as_string(defaultString);
auto exampleFloat = tensorMap.metadata().get("example.floatKey").as_float(defaultFloat);
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
