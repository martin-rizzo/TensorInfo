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
    'main.cpp',
    'messages.cpp',
)

executable(
    'lstensors',
    sources,
    dependencies : tensorinfo_static_dep,
    install      : false,
#   install      : true,
#   install_dir  : tools_dir_fullpath / 'bin'
)
