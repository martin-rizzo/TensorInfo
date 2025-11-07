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
#include <iostream>         // for std::cout
#include <tin/tensormap.h>  // for tin::TensorMap
#include "args.h"
#include "colors.h"
#include "message.h"
const char HELP[]=R"(
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

void help() {
    std::cout << HELP << std::endl;
}

//tin::TensorMap tensorMap;
//tensorMap.load( filepath );
//for( auto key: tensorMap.keys() ) {
//    std::cout << "key: " << key << " [" << tensorMap[key].shape() << "]" << std::endl;
//}
//tin::TensorMap tensorMap;
//tensorMap.load( filepathargv[1] );
//for( auto it = map.begin(); it != map.end(); ++it ) {
//    std::cout << "key: " << it->first << " [" << it->second.shape() << "]" << std::endl;
//}




int main(int argc, char* argv[]) {
    auto args = Args::from_main(argc, argv);

    if( args.no_color ) {
        Colors::instance().disable_colors();
    }

    if( args.help ) {
        help();
        return 0;
    }

    std::cout << args;


    // //// Get script name without extension
    // //std::string script_name = std::filesystem::path(argv[0]).filename().replace_extension("").string();
    
    
    // // determine what 
    // switch( args.command )
    // { 
    //     case Command::PRINT_TENSOR:
    //         if not args.file:
    //             fatal_error("No safetensors file provided. Please specify a file path")
    //         print_tensor_value(file_path=args.file, tensor_name=args.tensor)
    //         break;

    //     case Command::PRINT_METADATA:
    //         print_metadata(safedict, style=style)
    //         break;

    //     case Command::EXTRACT_THUMBNAIL:
    //         extract_thumbnail(safedict);
    //         break;

    //     default:
    //         print_tensor_names(safedict, max_depth=args.depth, style=style)
    //         break;

    // // Execute command
    // if (command == "strip") {
    //     if (input_file.empty()) {
    //         fatal_error("Missing input file for strip command");
    //     }
    //     if (output_file.empty()) {
    //         fatal_error("Missing output file for strip command");
    //     }
    //     strip_tensors(input_file, output_file);
    // } else {
    //     fatal_error("Invalid command: \"" + command + "\"", {"Use --help for usage."});
    // }
    
    // return 0;
}
