# File    : meson.build
# Purpose : Main meson build script for the "TensorInfo" project
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Nov 6, 2025
# Repo    : https://github.com/martin-rizzo/TensorInfo
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                                 TensorInfo
#   A C++ library for working with tensors & metadata in model checkpoints
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
#  Each subdirectory in this project should have its own "meson.build"
#  file which adds source files to the array `sources`.
#
#  Meson function reference: https://mesonbuild.com/Reference-manual_functions.html
#  Meson built-in options  : https://mesonbuild.com/Builtin-options.html
#  Meson for VS Code       : https://github.com/mesonbuild/vscode-meson
#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

project(
    'tensorinfo',            # Project name
    'cpp',                   # Primary language
    version: '0.1.0',        # Project version
    license: 'MIT',          # Project license
    default_options: [
        'cpp_std=c++20',     # Default C++ standard: C++20
        'buildtype=debug',   # Default build type  : debug
        'warning_level=3',   # Set warning level to 3 (high verbosity)
        'werror=true',       # Treat all warnings as errors
    ],
    meson_version: '>=1.1',  # Minimum Meson version required by this project
)

#-- C++ OPTIONS ------------------------------------------------------------#

# Suppressed warnings to be able to compile RapidJSON
add_project_arguments([
    '-Wno-unused-parameter',         #< suppress warnings about unused parameters
    '-Wno-deprecated-declarations',  #< suppress warnings about deprecated declarations
    '-Wno-class-memaccess'           #< suppress warnings about memcpy between objects
    ],
    language : 'cpp'
)

#-- DEPENDENCIES -----------------------------------------------------------#

# RapidJSON dependency (rapidjson_dep)
subdir('external' / 'rapidjson-1.1.0')

##=========================================================================##
## //////////////////////    TENSORINFO LIBRARY    /////////////////////// ##
##=========================================================================##
include_dirs = include_directories('include')
sources = [ ]
public_headers = [ ]
subdir( 'src'     / 'tin' )
subdir( 'include' / 'tin' )

# Install the public headers for the "TensorInfo" library
install_headers(
    public_headers,        # public headers to install
    preserve_path: true,   # if true, the directory structure must be preserved
    subdir       : 'tin',  # the subdirectory within the "include" dir
)

# Define the "TensorInfo" library as both static and shared libraries
tensorinfo_static = static_library(
    'tin',                                  # Name of the static library
    sources,                                # Sources for compiling the library
    include_directories: include_dirs,      # Include dirs for the library itself
    dependencies       : [ rapidjson_dep ], # Dependencies for the library
    install            : true,              # true = it should be installed when running 'meson install'
    install_dir        : 'lib',             # where to install it (relative to 'prefix')
)
tensorinfo_shared = shared_library(
    'tin',                                  # Name of the static library
    sources,                                # Sources for compiling the library
    include_directories: include_dirs,      # Include dirs for the library itself
    dependencies       : [ rapidjson_dep ], # Dependencies for the library
    install            : true,              # true = it should be installed when running 'meson install'
    install_dir        : 'lib',             # where to install it (relative to 'prefix')
)

# Create dependencies for each library type
tensorinfo_static_dep = declare_dependency(
    link_with          : tensorinfo_static,
    include_directories: include_dirs,
)
tensorinfo_shared_dep = declare_dependency(
    link_with          : tensorinfo_shared,
    include_directories: include_dirs,
)

