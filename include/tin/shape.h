/*
| File    : shape.h
| Purpose : A class representing the shape of a tensor.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 8, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_SHAPE_H_
#define TIN_SHAPE_H_
#include <tin/common.h>     // for Vector
#include <numeric>          // for std::accumulate
#include <initializer_list> // for std::initializer_list
#include <ostream>          // for std::ostream
#include <span>             // C++20: for std::span (view on contiguous data)
#include <compare>          // C++20: for the spaceship operator (<=>)
#include <format>           // C++20: for std::format()
namespace tin {


/**
 * Represents the shape of a tensor or multi-dimensional array.
 *
 * This class allows for intuitive definition and use of dimensions. It is an
 * immutable class, meaning that once constructed, its dimensions cannot be
 * modified. Utilizes some modern C++20 features.
 */
class Shape
{
// STL INTEGRATION
public:
    // These aliases are essential for making Shape compatible with some STL templates.
    using value_type      = unsigned int;      ///< Represents an individual dimension size.
    using size_type       = std::size_t;       ///< Alias for the size type used by containers.
    using difference_type = std::ptrdiff_t;    ///< Alias for the difference between iterators.

    // Since Shape is an immutable class (dimensions cannot be modified after
    // construction),references and pointers are constant.
    using reference       = const value_type&;
    using const_reference = const value_type&;
    using pointer         = const value_type*;
    using const_pointer   = const value_type*;

    // Iterators are also constant, as modifications to dimensions through them are not allowed.
    using iterator               = std::vector<value_type>::const_iterator;
    using const_iterator         = std::vector<value_type>::const_iterator;
    using reverse_iterator       = std::vector<value_type>::const_reverse_iterator;
    using const_reverse_iterator = std::vector<value_type>::const_reverse_iterator;


// CONSTRUCTION
public:


    /**
     * Default constructor that creates an empty shape (0 dimensions).
     */
    Shape() noexcept = default;


    /**
     * Constructs a copy of the given shape.
     */
    explicit Shape(const Shape& other) = default;


    /**
     * Constructs a move copy of the given shape.
     */
    explicit Shape(Shape&& other) noexcept = default;


    /**
     * Constructs a shape from an initializer list.
     * @param dims An initializer list representing the shape's dimensions.
     * @code
     * auto shape = Shape{2, 3, 4}; // creates a Shape with dimensions 2 x 3 x 4.
     * @endcode
     */
    explicit constexpr Shape(std::initializer_list<value_type> dims)
    : _dims(dims) { }


    /**
     * Constructs a shape from a vector of dimensions.
     * @param dims A `std::vector` containing the dimensions.
     * @code
     * std::vector<unsigned int> shape_dims = {2, 3, 4};
     * auto shape = Shape{shape_dims};
     * @endcode
     */
    explicit constexpr Shape(const std::vector<unsigned int>& dims)
    : _dims(dims) { }


    /**
     * Constructs a shape from a vector of dimensions (size_t variant).
     *
     * @param dims A `std::vector` containing the dimensions (as size_t elements).
     * @code
     * std::vector<size_t> shape_dims = {2, 3, 4};
     * auto shape = Shape{shape_dims};
     * @endcode
     */
    explicit constexpr Shape(const std::vector<size_t>& dims) {
        for( auto d: dims ) { _dims.push_back( static_cast<value_type>(d) );  }
    }


    /**
     * Constructs a shape from a span of dimensions.
     * @details Copies the provided data into the internal storage.
     * @param dims A `std::span` that provides a view of the dimensions to be copied.
     */
    explicit constexpr Shape(std::span<const value_type> dims)
    : _dims(dims.begin(), dims.end()) { }



// GETTING INFORMATION
public:

    /**
     * Returns the number of dimensions of the shape.
     *
     * This function acts as an alias for `size()`, providing the number
     * of dimensions of a shape, similar to obtaining the size of a std::vector.
     * It is intended to offer semantic clarity by specifically referring to 
     * "dimensions" in contexts where it may be more descriptive than simply 
     * using `size()`.
     *
     * @return int The number of dimensions.
     */
    [[nodiscard]] constexpr int ndim() const noexcept {
        return static_cast<int>(size());
    }

    /**
     * Returns the number of elements/dimensions (similar to std::vector::size()).
     *
     * This function provides the count of elements in the shape, which is
     * the same as what the `ndim()` function returns. It exists for consistency 
     * with the Standard Template Library (STL).
     *
     * @return size_type The number of elements/dimensions in the shape.
     */
    [[nodiscard]] constexpr size_type size() const noexcept {
        return _dims.size();
    }


    /**
     * Checks if the shape has no dimensions (is empty).
     *
     * This function returns true if there are no dimensions present in the
     * shape. It is similar to std::vector's `empty()` and provides a quick
     * way to verify whether any elements/dimensions exist.
     *
     * @return bool True if the shape has no dimensions; false otherwise.
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return _dims.empty();
    }

    /**
     * Calculates the total number of elements in a tensor with this shape.
     *
     * This function returns the total number of elements that would be
     * contained in a tensor if it had the current shape. If there are no
     * dimensions (i.e., the shape is empty), the function assumes it
     * represents a scalar or "null" form with one element, thus returning 1. 
     * Otherwise, it calculates the product of all dimensions.
     *
     * @return The total number of elements in the tensor.
     */
    [[nodiscard]] constexpr ULong numel() const noexcept {
        if (_dims.empty()) {
            return ULong{1}; // Representa un escalar o una "forma nula" con 1 elemento.
        }
        return std::accumulate(_dims.begin(), _dims.end(), ULong{1},
                               [](ULong acc, value_type dim) {
                                   return acc * dim;
                               });
    }


// ACCESSING DIMENSIONS
public:

    /**
     * Accesses the size of a dimension by its index using the subscript operator.
     * 
     * Provides constant-time access to individual dimensions. No bounds
     * checking is performed.
     * @param index Position index of the dimension whose size is desired
     * @return The size of the dimension at the given index.
     */
    [[nodiscard]] constexpr value_type operator[](size_type index) const noexcept {
        return _dims[index];
    }

    /**
     * Accesses the size of a dimension by its index with bounds checking.
     *
     * Provides safe access to individual dimensions with exception handling
     * for out-of-bound accesses.
     * @param index Position index of the dimension whose size is desired
     * @return The size of the dimension at the given index.
     */
    [[nodiscard]] constexpr value_type at(size_type index) const {
        return _dims.at(index);
    }

    /**
     * Access underlying data pointer of dimension sizes.
     *
     * Returns a constant pointer to the contiguous storage array used to
     * store the dimension sizes. This method is marked as `noexcept` and
     * can be useful for interfacing with C-style APIs or low-level operations
     * requiring direct access to memory.
     *
     * @return Pointer to the first element in the dimension size array.
     */
    [[nodiscard]] constexpr const_pointer data() const noexcept {
        return _dims.data();
    }


// ITERATORS
public:

    /**
     * Returns a (constant) iterator to the beginning of the dimensions.
     */
    [[nodiscard]] constexpr iterator begin() const noexcept { return _dims.cbegin(); }

   /**
     * Returns a (constant) iterator to the end of the dimensions.
     */
    [[nodiscard]] constexpr iterator end() const noexcept { return _dims.cend(); }

    /**
     * Returns an explicit constant iterator to the beginning of the dimensions.
     */
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept { return _dims.cbegin(); }

    /**
     * Returns an explicit constant iterator to the end of the dimensions.
     */
    [[nodiscard]] constexpr const_iterator cend() const noexcept { return _dims.cend(); }

    /**
     * Returns a (constant) reverse iterator to the beginning of the dimensions.
     */
    [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept { return _dims.crbegin(); }

    /**
     * Returns a (constant) reverse iterator to the end of the dimensions.
     */
    [[nodiscard]] constexpr reverse_iterator rend() const noexcept { return _dims.crend(); }

    /**
     * Returns an explicit constant reverse iterator to the beginning of the dimensions.
     */
    [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept { return _dims.crbegin(); }

    /**
     * Returns an explicit constant reverse iterator to the end of the dimensions.
     */
    [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept { return _dims.crend(); }


// DEBUGGING
public:
    [[nodiscard]] String to_string(StringView brackets = "[]", StringView separator = ", ") const;


// C++20
public:

    /**
     * Returns a span to the internal dimensions data.
     *
     * This method provides read-only access to the container's internal array
     * of dimensions. It returns a `std::span` object that represents a view
     * over the sequence of elements, allowing operations on this range without
     * copying or modifying the underlying data.
     *
     * @return A constant span representing a view over the internal dimensions data.
     */
    [[nodiscard]] constexpr std::span<const value_type> as_span() const noexcept {
        return std::span<const value_type>(_dims.data(), _dims.size());
    }

    /**
     * Comparison operator for shape objects using three-way comparison.
     *
     * This function implements the spaceship operator, which performs a
     * lexicographical comparison between two `Shape` instances. The operator
     * uses the default implementation, to determine the ordering relationship:
     * less than, equal to, or greater than.
     *
     * @param other Another Shape object for comparison against this one.
     * @return A three-way comparison result indicating whether `*this` is less
     *         than, equivalent to, or greater than `other`.
     */    
    [[nodiscard]] constexpr auto operator<=>(const Shape& other) const noexcept = default;

// IMPLEMENTATION
private:
    Vector<value_type> _dims; ///< Stores the dimensions of the shape.
};

} // namespace tin




/**
 * Overloads the insertion (<<) operator for outputting tin::Shape values to streams.
 *
 * This inline function allows objects of type `tin::Shape` to be directly
 * inserted into output streams such as `std::cout`.
 *
 * @param os    The output stream where the tin::Shape will be written.
 * @param shape The tin::Shape object to be inserted into the output stream.
 * @return A reference to 'os' for chaining.
 */
inline 
std::ostream& operator<<(std::ostream& os, const tin::Shape& shape) {
    return os << shape.to_string();
}


/**
 * Formatter specialization for tin::Shape objects.
 * 
 * Enables using C++20's formatting framework with tin::Shape objects. The
 * formatter converts the Shape object to its string representation and
 * formats it accordingly.
 * 
 * @code
 * #include <iostream>
 * #include <format>
 * #include <tin/shape.h>
 * 
 * int main() {
 *     tin::Shape shape = {10, 20};
 *     std::cout << std::format("The shape is {}", shape) << std::endl;
 *     // output = "The shape is [10, 20]"
 * }
 * @endcode
 * 
 */
template <>
struct std::formatter<tin::Shape> : std::formatter<std::string_view>
{
    auto format(const tin::Shape& shape, format_context& ctx) const {
        return std::formatter<std::string_view>::format(shape.to_string(), ctx);
    }
};



#endif // TIN_SHAPE_H_
