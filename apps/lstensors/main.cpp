/*
| File    : main.cpp
| Purpose : Main entry point for the `lstensors` command tool.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include "lstensors_args.h"
#include "lstensors_command.h"

//=========================================================================//
// //////////////////////////////// MAIN ///////////////////////////////// //
//=========================================================================//

int main(int argc, char* argv[]) {
    auto args    = LsTensorsArgs{argc, argv};
    auto command = LsTensorsCommand(args);
    return command.run();
}
