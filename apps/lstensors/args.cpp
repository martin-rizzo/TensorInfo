/*
| File    : args.cpp
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
#include "args.h"
#include "message.h"

/**
 * Parses command line arguments and returns a populated Args object.
 *
 * This function processes the command line arguments provided to the program,
 * distinguishing between flag-style options (prefixed with '-') and positional
 * arguments. Each argument is managed according to predefined rules.
 *
 * @param argc The number of command line arguments passed to the program.
 * @param argv An array of C strings representing the command line arguments.
 * @return A populated Args object containing parsed values from the
 *         command line arguments.
 */
Args
Args::from_main(int argc, char* argv[])
{
    Command     command{ Command::LIST_TENSORS };
    Format      format{ Format::HUMAN };
    std::string input_file{ };
    std::string prefix{ };
    std::string tensor_name{ };
    int         depth{ 0 };
    bool        no_color{ false };
    bool        help{ false };

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
            else if( arg=="-n" || arg=="--no-color" ) { no_color = true; }
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
    // return the parsed arguments
    return Args{
        command,
        format,
        input_file,
        prefix,
        tensor_name,
        depth,
        no_color,
        help
    };
}

/**
 * Overloads the insertion operator (<<) for printing Args objects to an output stream.
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
operator<<(std::ostream& os, const Args& args) {
    os << "Args:"                                              << std::endl;
    os << "  Command: "     << static_cast<int>(args.command)  << std::endl;
    os << "  Format: "      << static_cast<int>(args.format)   << std::endl;
    os << "  Input File: "  << args.input_file                 << std::endl;
    os << "  Prefix: "      << args.prefix                     << std::endl;
    os << "  Tensor Name: " << args.tensor_name                << std::endl;
    os << "  Depth: "       << args.depth                      << std::endl;
    os << "  No Color: "    << std::boolalpha << args.no_color << std::endl;
    os << "  Help: "        << std::boolalpha << args.help     << std::endl;
    return os;
}
