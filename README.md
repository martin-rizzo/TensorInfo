<div align="center">
  
# 🧭 TensorInfo (tin::)

![Phase 0](phase0.png)

</div>

**TensorInfo** is a lightweight C++ library designed to simplify the handling of tensor data stored in specialized file formats such as `.safetensors` and `.gguf`. It provides access to tensor metadata, shapes, and memory offsets, enabling seamless integration with machine learning checkpoints verifing and data processing workflows.

It includes utilities for inspecting safetensors/gguf files:
- `lstensors`: CLI tool to list tensors and metadata in safetensors and gguf files.

## Overview
> [!NOTE]
> This project is in "Phase 0", my initial brainstorming and planning stage. At this point, I'm exploring various approaches for implementation without a predetermined path. Consequently, features are still flexible and subject to change as the project develops further.

TensorInfo allows developers to load and inspect tensor maps from different file formats. The library currently includes two core classes:
- **TensorMap**: A high-level interface for managing tensor data and metadata.
- **TensorTree**: A hierarchical navigation system for exploring tensor structures within the loaded files.

## Key Features

- Load and parse tensor information from `.safetensors` and `.gguf` files.
- Access tensor metadata, including shapes, types, and byte offsets.
- Retrieve original file metadata.
- CLI utilities for quick analysis and exploration of tensor files.
- Simple design for easy integration with existing projects.

## Getting Started

To use TensorInfo, ensure your environment supports C++20 or later. Clone the repository and build using meson.
Contributions and feedback are welcome as the project evolves.

## License

Copyright (c) 2025 Martin Rizzo\
This project is licensed under the MIT license.\
See the ["LICENSE"](LICENSE) file for details.
