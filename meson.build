# File    : meson.build
# Purpose : Main meson build script for the "TensorInfo" project
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Nov 6, 2025
# Repo    : https://github.com/martin-rizzo/TensorInfo
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                                 TensorInfo
#   A C++ library for working with tensors & metadata in model checkpoints
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
    'tensorinfo',             # Project name
    'cpp',                   # Primary language
    version: '0.0.1',        # Project version
    license: 'MIT',          # Project license
    default_options: [       # Default options for the project
        'cpp_std=c++20',     # Use C++20 standard
        'warning_level=3',   # Set warning level to 3 (high verbosity)
        'werror=true',       # Treat all warnings as errors
        'buildtype=debug',   # Default build type: debug
    ],
    meson_version: '>=1.1',  # Minimum Meson version required by this project
)

#-- C++ OPTIONS ------------------------------------------------------------#

# Suppress warnings about unused parameters
add_project_arguments(['-Wno-unused-parameter'], language : 'cpp')


##=========================================================================##
## //////////////////////    TENSORINFO LIBRARY    /////////////////////// ##
##=========================================================================##
include_dirs = include_directories('include')
sources = [ ]
subdir('src' / 'tin' )

# Define the "TensorInfo" library as both static and shared libraries
tensorinfo_static = static_library(
    'tin',                              # Name of the static library
    sources,                            # Sources for compiling the library
    include_directories: include_dirs,  # Include dirs for the library itself
    # install: true                     # Install the library when running 'meson install'
)
tensorinfo_shared = shared_library(
    'tin',                              # Name of the static library
    sources,                            # Sources for compiling the library
    include_directories: include_dirs,  # Include dirs for the library itself
    # install: true                     # Install the library when running 'meson install'
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


##=========================================================================##
## ////////////////////////    TENSORINFO APPS    //////////////////////// ##
##=========================================================================##
subdir( 'apps' / 'lstensors' )
