# File    : meson.build
# Purpose : Main meson build script for RapidJSON C++ library
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Nov 10, 2025
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  Meson function reference: https://mesonbuild.com/Reference-manual_functions.html
#  Meson built-in options  : https://mesonbuild.com/Builtin-options.html
#  Meson for VS Code       : https://github.com/mesonbuild/vscode-meson
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

project(
    'rapidjson',            # Project name
    'cpp',                  # Primary language
    default_options: [
        'cpp_std=c++20',    # Default C++ standard: C++20
        'buildtype=debug',  # Default build type  : debug
        'warning_level=3',  # Set warning level to 3 (high verbosity)
        'werror=true',      # Treat all warnings as errors
    ],
    version       : '1.1.0',         # Project version
    license       : 'MIT',           # Project license
    meson_version : '>=1.1',         # Minimum Meson version required by this project
)

#=============================================================================
# RapidJSON is a header-only library, requiring no separate source
# files or external linking for dependency configuration.
#=============================================================================

# Include directories of Mr.Diffusion library
include_dirs = include_directories('include')

# Declare a dependency so that other targets, like an app, can use it.
rapidjson_dep = declare_dependency(
    include_directories: include_dirs,
    #link_with         : <nothing>,
)
