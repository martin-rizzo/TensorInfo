/*
| File    : message.h
| Purpose : Simple class to display messages to the user in the console.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <string_view> // for std::string_view
#include <vector>      // for std::vector
#include <iostream>    // for std::cerr
#include "message.h"
#include "colors.h" 


/**
 * Displays a warning message to the console.
 *
 * This method is used to output messages that serve as warnings. These messages
 * indicate potential problems or non-critical information, allowing execution
 * to continue.
 * @param message The content of the warning message.
 */
void
Message::warning(std::string_view message) {
    auto c = Colors::instance();
    std::cerr << c.cyan() << "[WARNING]" << c.reset() << " " << message << std::endl;
}

/**
 * Displays an error message to the console.
 *
 * This method is used to output messages indicating errors encountered during
 * execution. These errors are important but do not necessarily require
 * immediate termination of the program.
 * @param message The content of the error message.
 */
void
Message::error(std::string_view message) {
    auto c = Colors::instance();
    std::cerr << c.red() << "[ERROR]" << c.reset() << " " << message << std::endl;
}

/**
 * Displays a fatal error message to the console and exits the program.
 *
 * This method is used for displaying critical errors that require immediate
 * termination of the application. It first outputs a primary error message
 * followed by any additional informative messages provided.
 * 
 * The program execution is terminated using `std::exit` with ??? 
 *
 * @param message             The main content of the fatal error message.
 * @param informativeMessages A vector containing additional texts that provide
 *                            more context or details about the fatal error.
 * @param exitCode            ??
 */

/**
 * Displays a fatal error message to the console and exits the program.
 *
 * This method is used for displaying critical errors that necessitate the
 * immediate termination of the application. It begins by outputting a primary
 * error message, followed by any supplementary informative messages provided
 * as additional context.
 * 
 * After displaying these messages, it terminates the program execution using
 * `std::exit`.
 *
 * @param message             The main content of the fatal error message.
 * @param informativeMessages An optional vector containing additional texts that
 *                            provide more context or details about the fatal error.
 * @param exitCode            An optional integer representing the termination
 *                            status of the program. 
 */
void
Message::fatal_error(std::string_view message,
                     const std::vector<std::string_view>& informativeMessages, // = {},
                     int exitCode // = 1
){
    error(message);
    
    // Print additional messages if any
    auto c = Colors::instance();
    for (const auto& infoMessage : informativeMessages) {
        std::cerr << " " << c.cyan() << "\xF0\x9F\x9B\x88 " << infoMessage << c.reset() << std::endl;
    }
    std::exit( exitCode>=1 ? exitCode : 1 );
}
