# File    : meson.build
# Purpose : Declares the sources and subdirs for this folder
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Nov 6, 2025
# Repo    : https://github.com/martin-rizzo/TensorInfo
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

sources = [ ]

#subdir('<none>')
sources += files(
    'colors.cpp',
    'lstensors_args.cpp',
    'lstensors_command.cpp',
    'main.cpp',
    'message.cpp',
)

executable(
    'lstensors',
    sources,
    dependencies : tensorinfo_static_dep,
    install      : true,   # true = it should be installed when running 'meson install'
    install_dir  : 'bin',  # where to install it (relative to 'prefix')
)
