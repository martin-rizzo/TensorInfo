/*
| File    : <tin/sortby.h>
| Purpose : Enumerate the different sorting criteria for tensors.
| Author  : Martin Rizzo | <martinrizzo@gmail.com>
| Date    : Nov 15, 2025
| Repo    : https://github.com/martin-rizzo/TensorInfo
| License : MIT
|- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
|                                 TensorInfo
|   A C++ library for working with tensors & metadata in model checkpoints
\_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _*/
#pragma once
#ifndef TIN_SORTBY_H_
#define TIN_SORTBY_H_
namespace tin {


/**
 * Enumerates the different sorting criteria for tensors.
 *
 * This enum is utilized to determine how tensor information should be sorted,
 * it is used in methods such as:
 *   `std::vector<TensorInfo> TensorMap::collect_tensors(SortBy);`
 * 
 * @see TensorMap::collect_tensors(SortBy)
 */
enum class SortBy {
    NONE,     ///< No sorting is applied to the tensors, very fast.
    DTYPE,    ///< Sorts tensors based on data type (e.g., Q2_K, F32, F64).
    BYTES,    ///< Sorts tensors by the number of raw bytes they occupy in storage.
    NUMEL,    ///< Sorts tensors according to the total number of elements, as reported by `TensorInfo::numel()`.

    /**
     * `NAME` sorts tensor names by interpreting numeric sub-indices within
     * their names logically rather than alphabetically. For instance, in
     * tensor names like "model.layers.X.input_layernorm.weight", where 'X'
     * denotes a layer number, this method prioritizes logical sequence.
     * 
     * In contrast to alphabetical sorting, where "layer.10.input" might precede
     * "layer.9.input" due to character comparison, `NAME` ensures that tensors
     * are arranged based on their actual numerical values. This results in a
     * sequence such as "layer.8", "layer.9", "layer.10", and so forth,
     * reflecting the true numeric progression.
     */     
     NAME
};


}      // tin namespace
#endif // TIN_SORTBY_H_