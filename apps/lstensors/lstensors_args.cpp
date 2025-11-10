/*
| File    : lstensors_args.h
| Purpose : The arguments of the `lstensors` command line
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 7, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <string>
#include "lstensors_args.h"
#include "message.h"

/**
 * Constructs a new LsTensorsArgs object by parsing command line arguments.
 *
 * This function processes the command line arguments provided to the program,
 * distinguishing between flag-style options (prefixed with '-') and positional
 * arguments. Each argument is managed according to predefined rules.
 *
 * @param argc The number of command line arguments passed to the program.
 * @param argv An array of C strings representing the command line arguments.
 */
LsTensorsArgs::LsTensorsArgs(int argc, char* argv[])
{
    for( int i=1 ; i < argc ; ++i ) {
        std::string arg{ argv[i] };

        // parse the flags arguments
        if( arg.starts_with('-') ) {
            std::string nextarg{ (i+i) < argc ? argv[i+1] : "" };
            if     ( arg=="-p" || arg=="--prefix"   ) { prefix = nextarg; }
            else if( arg=="-d" || arg=="--depth"    ) { depth  = std::stoi(nextarg); }
            else if( arg=="-t" || arg=="--tensor"   ) { command = Command::PRINT_TENSOR; tensor_name = nextarg; }
            else if( arg=="-m" || arg=="--metadata" ) { command = Command::PRINT_METADATA; }
            else if(              arg=="--thumbnail") { command = Command::EXTRACT_THUMBNAIL; }
            else if( arg=="-u" || arg=="--human"    ) { format = Format::HUMAN; }
            else if( arg=="-b" || arg=="--plain"    ) { format = Format::PLAIN; }
            else if( arg=="-j" || arg=="--json"     ) { format = Format::JSON; }
            else if( arg=="-n" || arg=="--no-color" ) { use_color = UseColor::NEVER; }
            else if( arg=="-h" || arg=="--help"     ) { help = true; }
            else {
                Message::fatal_error("Unknown argument: " + arg, {
                    "Try `lstensors --help` for more information."
                });
            }
        }
        // parse the positional arguments (not starting with '-')
        else {
            if( input_file.empty() ) { input_file = arg;  }
            else {
                Message::fatal_error("Too many files specified.", {
                    "You can only specify one file."
                });
            }
        }
    }
}

/**
 * Overloads the insertion operator (<<) for printing LsTensorsArgs objects to an output stream.
 *
 * This function provides a way to print the contents of an Args object in a 
 * human-readable format. It outputs each member variable with its name and
 * value to the specified output stream, facilitating debugging or logging.
 *
 * @param os   The output stream where the Args data will be printed.
 * @param args The Args object being printed to the stream.
 * @return A reference `os` for chaining.
 */
std::ostream&
operator<<(std::ostream& os, const LsTensorsArgs& args) {
    os << "Args:"                                        << std::endl;
    os << "  command: "     << to_string(args.command)   << std::endl;
    os << "  format: "      << to_string(args.format)    << std::endl;
    os << "  input_file: "  << args.input_file           << std::endl;
    os << "  prefix: "      << args.prefix               << std::endl;
    os << "  tensor_name: " << args.tensor_name          << std::endl;
    os << "  depth: "       << args.depth                << std::endl;
    os << "  use_color: "   << to_string(args.use_color) << std::endl;
    os << "  Help: "        << to_string(args.help)      << std::endl;
    return os;
}
