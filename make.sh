#!/usr/bin/env bash
# File    : make.sh
# Purpose : Script to illustrate how to compile the "TensorInfo" project
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Nov 6, 2025
# Repo    : https://github.com/martin-rizzo/TensorInfo
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                                 TensorInfo
#   A C++ library for working with tensors & metadata in model checkpoints
#
#     Copyright (c) 2025 Martin Rizzo
#
#     Permission is hereby granted, free of charge, to any person obtaining
#     a copy of this software and associated documentation files (the
#     "Software"), to deal in the Software without restriction, including
#     without limitation the rights to use, copy, modify, merge, publish,
#     distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so, subject to
#     the following conditions:
#
#     The above copyright notice and this permission notice shall be
#     included in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}" .sh)         # script name without extension
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")") # script directory
PROJECT_DIR=$SCRIPT_DIR                                 # project directory
HELP="
Usage: ./$SCRIPT_NAME.sh [COMMAND] [OPTIONS]

  Allows you to compile and manage the 'TensorInfo' project in Linux environments.

  Available command:
    build        Build the project (default)
    rebuild      Rebuild the project from scratch
    clean        Clean the build directory
    library      Build only the library (libtin.a)
    tools        Build the command-line tools

  Available options:
    -l, --error-limit [N]  Maximum number of errors to show (default 30 if 'N' is empty)
    -n, --no-color         Disable color output.
    -h, --help             Show this help message and exit.

  Examples:
    $SCRIPT_NAME clean
    $SCRIPT_NAME --no-color build
"

# ANSI escape codes for colored terminal output
RED='\e[91m'; YELLOW='\e[93m'; GREEN='\e[92m'; CYAN='\e[96m'; RESET='\e[0m'
disable_color() { RED=''; YELLOW=''; GREEN=''; CYAN=''; RESET=''; }

#============================= ERROR MESSAGES ==============================#

# Display help message
help() { echo "${HELP}"; }

# Display a warning message
warning() { echo -e "\n${CYAN}[${YELLOW}WARNING${CYAN}]${RESET} $1" >&2; }

# Display an error message
error() { echo -e "\n${CYAN}[${RED}ERROR${CYAN}]${RESET} $1" >&2; }

# Displays a fatal error message and exits the script
fatal_error() {
    error "$1"; shift
    while [[ $# -gt 0 ]]; do
        echo -e " ${CYAN}\xF0\x9F\x9B\x88 $1${RESET}" >&2
        shift
    done
    echo; exit 1
}

#================================ COMMANDS =================================#


# Builds the TensorInfo project using the Meson build system.
#
# Usage:
#   build_project [ERROR_LIMIT]
#
# Parameters:
#   [ERROR_LIMIT]: An optional parameter that specifies the maximum number of
#                  error messages to display. If not provided, all errors will
#                  be shown during compilation.
#
# Requirements:
#   - Meson and Ninja build systems must be installed on the system.
#
build_project() {
    local error_limit=$1  #< (optional) must be empty or a positive integer

    # make sure we are at the root of the project directory
    cd "$PROJECT_DIR" || fatal_error "Failed to change directory."

    # this is a small hack to show only the first compilation errors up to 'error_limit'
    # it uses `grep` to find error messages and then `head` to limit the number of lines shown
    if [[ "${error_limit}" ]]; then
        meson setup      builddir
        meson install -C builddir 2>&1 | grep -E "error:|fatal error:" --color="always" | head -n "${error_limit}"
        return 0
    fi
    

    # this is the recommended way to compile the project
    meson setup      builddir
    meson install -C builddir

}


#
# Cleans the project by removing the build directory.
#
clean_project() {
    # make sure we are at the root of the project directory
    cd "$PROJECT_DIR"   || { fatal_error "Failed to change directory."; }
    [[ -d 'builddir' ]] || { warning "No build directory found. Nothing to clean."; return 0; }

    rm -rf 'builddir'
    echo "Build directory cleaned."
}


# #===========================================================================#
# #////////////////////////////////// MAIN ///////////////////////////////////#
# #===========================================================================#
main() {

    local no_color=false   # no color mode flag
    local help=false       # help mode flag
    local command="build"  # default command to execute
    local error_limit=''   # no error limit

    while [[ $# -gt 0 ]]; do
        if [[ $1 == -* ]]; then
            case $1 in
                -l|--error-limit)
                    if [[ "$2" =~ ^[0-9]+$ ]]; then
                        error_limit="$2"; shift
                    else
                        error_limit=30 #< default error limit if -l without argument
                    fi
                ;;
                -n|--nc|--no-color) no_color=true           ;;
                -h|--help)          help=true               ;;
                *)
                    fatal_error "Invalid option: \"$1\"" "Use --help for usage." ;;
            esac
        else
            command=$1;
        fi
        shift
    done

    # disable color output if --no-color option is provided
    if [[ $no_color == true ]]; then disable_color; fi

    # display help message and exit if --help option is provided
    if [[ $help == true ]]; then help; exit 0; fi

    # execute the specified command
    case $command in
        build|all)  build_project "$error_limit" ; exit 0 ;;
        clean)      clean_project                ; exit 0 ;;
        rebuild)    clean_project; build_project ; exit 0 ;;
        library)    fatal_error "Not implemented yet." ;;
        tools)      fatal_error "Not implemented yet." ;;
        *)          fatal_error "Invalid command: \"$command\"" "Use --help for usage." ;;
    esac

}
main "$@"
