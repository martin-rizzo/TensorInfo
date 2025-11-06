<div align="center">

# 🗺️ TensorMap (tm):<br/> A C++ library for inspecting tensor metadata in .safetensors and .gguf files.

</div>

TensorMap is a lightweight C++ library designed to simplify the handling of tensor data stored in specialized file formats such as `.safetensors` and `.gguf`. It provides tools to efficiently access tensor metadata, shapes, and memory offsets, enabling seamless integration with machine learning models and data processing workflows.

It includes utilities for viewing safetensors/gguf files:
- `lsttensors`: CLI tool to list tensors and metadata in `.safetensors` and `.gguf` files.

## Overview

TensorMap allows developers to load and inspect tensor maps from disk, leveraging structured metadata and direct memory access. The library currently includes two core classes:
- **TensorMap** : A high-level interface for managing tensor data and metadata.
- **TensorTree**: A hierarchical navigation system for exploring tensor structures within the loaded files.

note: This project is in its early development phase

## Key Features  

- Load and parse tensor maps from `.safetensors` and `.gguf` files.  
- Access tensor metadata, including shapes, types, and byte offsets.  
- Retrieve original file metadata.
- CLI utilities for quick analysis and exploration of tensor files.  
- Simple design for easy integration with existing projects.

## Getting Started

To use TensorMap, ensure your environment supports C++20 or later. Clone the repository and build using meson.
Contributions and feedback are welcome as the project evolves.

## License

Copyright (c) 2025 Martin Rizzo  
This project is licensed under the MIT license.  
See the ["LICENSE"](LICENSE) file for details.
