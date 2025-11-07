#include <tin/tensormap.h>
#include "args.h"
#include "colors.h"
#include "messages.h"
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






int main(int argc, char* argv[])
{
    Message::fatal_error("This command is not implemented yet.", {
        "More info at https://github.com/martin-rizzo/TensorInfo",
        "This is a information message."
    });

    // //// Get script name without extension
    // //std::string script_name = std::filesystem::path(argv[0]).filename().replace_extension("").string();
    
    // // parse arguments
    // auto args = Args{ argc, argv };
    
    // // disable colors if requested
    // if( args.no_color ) { Colors::global().disable_colors(); }
    
    // // Display help if requested
    // if (help_mode) {
    //     help();
    //     return 0;
    // }
    
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