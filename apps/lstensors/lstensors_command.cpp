/*
| File    : lstensors_command.cpp
| Purpose : The `lstensors` command line tool.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 7, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <tin/tensormap.h>
#include "colors.h"
#include "message.h"
#include "lstensors_command.h"
static const char HELP[]=R"(
Usage: lstensors [OPTIONS] file

  Allows you to compile and manage the OpenDiffusion project in Linux.

  OPTIONS:
    -p, --prefix <PREFIX>  Filter the tensor names by a prefix to display only matching tensors
    -d, --depth <DEPTH>    Specify the depth level of the hierarchical index to display 
    -t, --tensor <TENSOR>  Show the value of a specific tensor (e.g., 'model.layer.1.bias')
    -m, --metadata         Print only the metadata information from the file
    --thumbnail            Extract the thumbnail from the .safetensors file and save it as a .jpg image

  Output formats:
    -u, --human            Output in a human-readable format with clear formatting (default)
    -b, --plain            Output in a plain, easily parseable format for scripts or tools
    -j, --json             Output data in JSON format when available

    -n, --no-color         Disable color output.
    -h, --help             Show this help message and exit.

  Examples:
    lstensors --prefix model.layer.1.bias 'checkpoint.safetensors'
    lstensors --no-color 'checkpoint.safetensors'
)";

#ifdef _WIN32
    inline bool is_terminal_output() { return true; }
#else
#include <unistd.h> // for "::isatty()" and STDOUT_FILENO
    inline bool is_terminal_output() { return ::isatty(STDOUT_FILENO) != 0; }
#endif

using namespace tin;

//============================= CONSTRUCTION ==============================//

LsTensorsCommand::LsTensorsCommand(const LsTensorsArgs& args
): _args(args)
{}


//================================ RUNNING ================================//

int
LsTensorsCommand::run()
{
    ReadError readError;

    // determine if we should use color output
    const bool use_color = _args.use_color == UseColor::ALWAYS ||
                          (_args.use_color == UseColor::AUTO && is_terminal_output());
    if( !use_color ) {
        Colors::instance().disable_colors();
    }

    // if help was requested, show the help message and exit
    if( _args.help ) { print_help(); return 0; }

    // if the user didn't provide any file, show an error message and exit
    if( _args.input_file.empty() ) {
        Message::fatal_error("No file provided. Please specify a .safetensors or .gguf file.", {
            "To get help on how to use this tool, run: lstensors --help"
        });
    }

    // load the checkpoint file
    const auto tensorMap = TensorMap::from_file( _args.input_file, readError );
    if( readError != ReadError::None ) { fatal_read_error( readError ); }

    // print the names of all tensors in the file
    for( auto it: tensorMap ) {
        std::cout << it.first << " : " << it.second.dtype() << std::endl;
    }

    return 0;
}


//================================ HELPERS ================================//

void
LsTensorsCommand::print_help() {
    std::cout << HELP << std::endl;
}

void
LsTensorsCommand::fatal_read_error(ReadError readError) {
    switch( readError )
    {
        case ReadError::FileNotFound:
            Message::fatal_error("File not found.");

        case ReadError::InvalidFormat:
            Message::fatal_error("This is probably not a valid .safetensors or .gguf file.");

        case ReadError::UnsupportedVersion:
            Message::fatal_error("The file may be from an older or newer version of the format that this tool does not support.");

        case ReadError::HeaderTooLarge:
            Message::fatal_error("The file header may be corrupted, incomplete, or have other issues that prevent it from being read correctly.");

        case ReadError::MemoryAllocationFailed:
            Message::fatal_error("There may not be enough memory available to read this file, or it is corrupted in a way that prevents allocation of enough memory.");

#if 0
        case ReadError::InvalidMagicNumber: { Message::fatal_error("Invalid magic number. This is probably not a valid .safetensors or .gguf file."); } break;
        case ReadError::InvalidHeaderSize: { Message::fatal_error("Invalid header size. This is probably not a valid .safetensors or .gguf file."); } break;
        case ReadError::InvalidTensorName: { Message::fatal_error("Invalid tensor name. The tensor name may be too long, contain invalid characters, or have other issues."); } break;
        case ReadError::InvalidTensorDataSize: { Message::fatal_error("Invalid tensor data size. This is probably not a valid .safetensors or .gguf file."); } break;
        case ReadError::InvalidMetadata: { Message::fatal_error("Invalid metadata. The metadata may be corrupted, incomplete, or have other issues."); } break;
        case ReadError::UnsupportedDataType: { Message::fatal_error("Unsupported data type. This tool does not support the specified data type in this file."); } break;
        case ReadError::FileTooLarge: { Message::fatal_error("File too large. The file may be corrupted or have other issues that prevent it from being read correctly."); } break;
        case ReadError::UnknownError: { Message::fatal_error("Unknown error. An unknown error occurred while reading the file."); } break;
#endif
        default:
            Message::fatal_error("An unknown error occurred while reading the file.");
    }
}
