/*
| File    : colors.h
| Purpose : A class for handling terminal colors.
|           This class is a singleton and can be accessed through the instance() method.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 6, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef COLORS_H_
#define COLORS_H_
#include <string_view>


/**
 * ANSI color codes manager for terminal output
 * 
 * This class provides a thread-safe singleton instance for managing ANSI color
 * codes for terminal output. It supports disabling colors and provides various
 * color codes for different text formatting.
 * 
 * Example usage:
 * @code{.cpp}
 * #include <iostream>
 *
 * int main() {
 *     auto c = Colors::instance();
 *
 *     std::cout << c.red()   << "This is red text" << c.reset()   << std::endl;
 *     std::cout << c.green() << "This is green text" << c.reset() << std::endl;
 *
 *     // Disable color output
 *     c.disable_colors();
 *     std::cout << c.red() << "This will not be colored" << c.reset() << std::endl;
 *
 *     return 0;
 * }
 * @endcode
 */
class Colors
{
public:
     [[nodiscard]]
     static Colors& instance() noexcept;


// CONSTRUCTION/ASSIGNMENT
public:
    /** Copy constructor */
    Colors(const Colors&) = default;

    /** Move constructor */
    Colors(Colors&&) noexcept = default;

    /** Copy assignment operator  */
    Colors& operator=(const Colors&) = default;

    /** Move assignment operator  */
    Colors& operator=(Colors&&) noexcept = default;

    ~Colors() = default;


// GETTING COLORS
public:

    /** Returns the red color code */
    [[nodiscard]] std::string_view red() const noexcept { return _red; }

    /** Returns the yellow color code */
    [[nodiscard]] std::string_view yellow() const noexcept { return _yellow; }

    /** Returns the green color code */
    [[nodiscard]] std::string_view green() const noexcept { return _green; }

    /** Returns the cyan color code */
    [[nodiscard]] std::string_view cyan() const noexcept { return _cyan; }

    /** Returns the color code that resets the color */
    [[nodiscard]] std::string_view reset() const noexcept { return _reset; }

    [[nodiscard]] std::string_view get_color_code(const std::string_view& colorName) noexcept;


// CONTROL COLORS
public:
    void disable_colors() noexcept;
    [[nodiscard]] bool are_colors_enabled() const noexcept;


// IMPLEMENTATION
private:
    Colors() = default;
private:
    const char* _red   { "\x1b[91m" };
    const char* _yellow{ "\x1b[93m" };
    const char* _green { "\x1b[92m" };
    const char* _cyan  { "\x1b[96m" };
    const char* _reset { "\x1b[0m"  };
};


#endif // COLORS_H_
