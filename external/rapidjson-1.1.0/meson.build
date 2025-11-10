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

#=============================================================================
# RapidJSON is a header-only library, requiring no separate source
# files or external linking for dependency configuration.
#=============================================================================

## Scan source directory for C++ source files and add them to `lib_sources`
#lib_sources = [ ]
#subdir('source')

# Include directories of Mr.Diffusion library
include_dirs = include_directories('include')

# Declare a dependency so that other targets, like an app, can use it.
rapidjson_dep = declare_dependency(
    include_directories: include_dirs,
    #link_with         : <nothing>,
)
