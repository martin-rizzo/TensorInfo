/*
| File    : colors.cpp
| Purpose : A class for handling terminal colors.
|           This class is a singleton and can be accessed through the global() method.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#include <mutex>   // for std::once_flag
#include <memory>  // for std::unique_ptr
#include "colors.h"


/**
 * Get the singleton instance of Colors
 * @return Reference to the singleton instance
 * @note This method is thread-safe (because if we're going to do it, let's get it right!)
 */
Colors& Colors::instance() noexcept {
    static std::once_flag          flag;
    static std::unique_ptr<Colors> instancePtr;
    std::call_once(flag, []() {
        instancePtr = std::unique_ptr<Colors>( new Colors() );
    });
    return *instancePtr;
}

/**
 * Get color code by name
 * @param colorName Name of the color (red, yellow, green, cyan)
 * @return The ANSI code for the specified color or "reset code" if color name is not found.
 */
std::string_view
Colors::get_color_code(const std::string_view& colorName) noexcept {
    if (colorName == "red"   ) { return red();    }
    if (colorName == "yellow") { return yellow(); }
    if (colorName == "green" ) { return green();  }
    if (colorName == "cyan"  ) { return cyan();   }
    return reset();
}

/**
 * Disable all colors
 */
void
Colors::disable() noexcept {
    _red = _yellow = _green = _cyan = _reset = "";
}


/**
 * Check if color codes are enabled
 * @return true if colors are enabled, false otherwise
 */
bool
Colors::is_enabled() const noexcept {
    return _red[0] != '\0';
}
