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

This section outlines how to interact with metadata using `TensorMap`.

### Accessing Metadata Values

Retrieve a specific metadata value using the `get` method from `tensorMap.metadata()`, specifying the key name. If needed, provide a default value when the key is missing.

**Example:**
```cpp
auto contextLength = tensorMap.metadata().get("llama.context_length").as_integer(default_value);
```
This retrieves an integer value for `"llama.context_length"`, using `default_value` if absent.

### Modifying Metadata Values

Update or set metadata values with the `set_integer` method, specifying the key and new value.

**Example:**
```cpp
tensorMap.metadata().set_integer("llama.context_length", 8 * 1024);
```
This sets `"llama.context_length"` to `8192`.

### Enumerating Metadata Values

Iterate through all metadata entries using a loop, accessing each as a key-value pair.

**Example:**
```cpp
std::cout << "Metadata:" << std::endl;
for (auto& it : tensorMap.metadata()) {
    std::cout << "  " << it.first << ": " << it.second.to_string() << std::endl;
}
```
This prints each metadata key and its string representation.

In summary, `tensorMap.metadata()` provides efficient tools for accessing and modifying checkpoint metadata, crucial for model configuration and tuning.


## License

Copyright (c) 2025 Martin Rizzo\
This project is licensed under the MIT license.\
See the ["LICENSE"](LICENSE) file for details.
