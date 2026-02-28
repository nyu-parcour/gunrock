#pragma once

#include <string>

namespace gunrock {
namespace metal {
namespace shaders {

inline const std::string& get_all_shaders() {
  static const std::string source = R"(
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ============================================================
// Primitives: fill, for_each, exclusive_scan, reduce, copy_if
// ============================================================

kernel void fill_int(
    device int* output       [[buffer(0)]],
    constant uint& count     [[buffer(1)]],
    constant int& value      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count)
        output[tid] = value;
}

kernel void fill_uint(
    device uint* output      [[buffer(0)]],
    constant uint& count     [[buffer(1)]],
    constant uint& value     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count)
        output[tid] = value;
}

kernel void fill_float(
    device float* output     [[buffer(0)]],
    constant uint& count     [[buffer(1)]],
    constant float& value    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count)
        output[tid] = value;
}

// Per-threadgroup scan with partial sums output.
// Each threadgroup scans up to SCAN_BLOCK_SIZE elements and writes
// its total to partial_sums[threadgroup_id].
constant uint SCAN_BLOCK_SIZE = 256;

kernel void scan_per_block(
    device const uint* input       [[buffer(0)]],
    device uint* output            [[buffer(1)]],
    device uint* partial_sums      [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    uint tid   [[thread_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint gid   [[threadgroup_position_in_grid]],
    uint tg_sz [[threads_per_threadgroup]])
{
    threadgroup uint shared_data[256];

    uint val = (tid < count) ? input[tid] : 0;
    shared_data[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce)
    for (uint stride = 1; stride < tg_sz; stride *= 2) {
        uint idx = (lid + 1) * stride * 2 - 1;
        if (idx < tg_sz)
            shared_data[idx] += shared_data[idx - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store block total and clear last element for exclusive scan
    if (lid == tg_sz - 1) {
        partial_sums[gid] = shared_data[lid];
        shared_data[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint stride = tg_sz / 2; stride >= 1; stride /= 2) {
        uint idx = (lid + 1) * stride * 2 - 1;
        if (idx < tg_sz) {
            uint temp = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < count)
        output[tid] = shared_data[lid];
}

// Propagate partial sums from block-level scan to produce global result.
kernel void scan_propagate(
    device uint* output            [[buffer(0)]],
    device const uint* partial_sums [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint tid   [[thread_position_in_grid]],
    uint gid   [[threadgroup_position_in_grid]])
{
    if (gid > 0 && tid < count)
        output[tid] += partial_sums[gid];
}

// Single-threadgroup scan for the partial sums array.
// Handles up to 256 elements (enough for up to 256*256 = 65536 input elements).
kernel void scan_partial_sums(
    device uint* partial_sums      [[buffer(0)]],
    constant uint& count           [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup uint shared_data[256];

    uint val = (lid < count) ? partial_sums[lid] : 0;
    shared_data[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < 256; stride *= 2) {
        uint idx = (lid + 1) * stride * 2 - 1;
        if (idx < 256)
            shared_data[idx] += shared_data[idx - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 255)
        shared_data[255] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride >= 1; stride /= 2) {
        uint idx = (lid + 1) * stride * 2 - 1;
        if (idx < 256) {
            uint temp = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid < count)
        partial_sums[lid] = shared_data[lid];
}

// Tree-based reduction: each threadgroup reduces its portion and atomically
// accumulates into a global result.
kernel void reduce_sum_uint(
    device const uint* input       [[buffer(0)]],
    device atomic_uint* result     [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint tid   [[thread_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    threadgroup uint shared_data[256];

    shared_data[lid] = (tid < count) ? input[tid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_sz / 2; stride > 0; stride /= 2) {
        if (lid < stride)
            shared_data[lid] += shared_data[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        atomic_fetch_add_explicit(result, shared_data[0], memory_order_relaxed);
}

kernel void reduce_sum_int(
    device const int* input        [[buffer(0)]],
    device atomic_int* result      [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint tid   [[thread_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    threadgroup int shared_data[256];

    shared_data[lid] = (tid < count) ? input[tid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_sz / 2; stride > 0; stride /= 2) {
        if (lid < stride)
            shared_data[lid] += shared_data[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        atomic_fetch_add_explicit(result, shared_data[0], memory_order_relaxed);
}

// Mark valid elements for stream compaction (invalid = -1 sentinel).
kernel void mark_valid(
    device const int* input        [[buffer(0)]],
    device uint* marks             [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count)
        marks[tid] = (input[tid] >= 0) ? 1 : 0;
}

// Scatter valid elements using pre-computed offsets from exclusive scan.
kernel void scatter_valid(
    device const int* input        [[buffer(0)]],
    device int* output             [[buffer(1)]],
    device const uint* marks       [[buffer(2)]],
    device const uint* offsets     [[buffer(3)]],
    constant uint& count           [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count && marks[tid] == 1)
        output[offsets[tid]] = input[tid];
}

// ============================================================
// Graph Operators
// ============================================================

// Thread-mapped advance: one thread per frontier vertex, each thread
// iterates over its neighbors.
// The advance_op parameter determines what to write into output_frontier;
// here we use a generic version that just outputs the neighbor.
kernel void advance_thread_mapped(
    device const uint* row_offsets       [[buffer(0)]],
    device const int*  column_indices    [[buffer(1)]],
    device const int*  input_frontier    [[buffer(2)]],
    device int*        output_frontier   [[buffer(3)]],
    device const uint* segment_offsets   [[buffer(4)]],
    constant uint& frontier_size         [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= frontier_size) return;

    int v = input_frontier[tid];
    if (v < 0) return;

    uint start = row_offsets[v];
    uint end   = row_offsets[v + 1];
    uint out_base = segment_offsets[tid];

    for (uint e = start; e < end; ++e) {
        output_frontier[out_base + (e - start)] = column_indices[e];
    }
}

// Compute per-vertex output sizes (degrees) for advance output allocation.
kernel void compute_output_offsets(
    device const uint* row_offsets       [[buffer(0)]],
    device const int*  input_frontier    [[buffer(1)]],
    device uint* output_sizes            [[buffer(2)]],
    constant uint& frontier_size         [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= frontier_size) return;

    int v = input_frontier[tid];
    if (v < 0) {
        output_sizes[tid] = 0;
        return;
    }
    output_sizes[tid] = row_offsets[v + 1] - row_offsets[v];
}

// ============================================================
// BFS Algorithm Kernels
// ============================================================

// BFS advance: expand frontier, update distances atomically.
kernel void bfs_advance(
    device const uint* row_offsets       [[buffer(0)]],
    device const int*  column_indices    [[buffer(1)]],
    device const int*  input_frontier    [[buffer(2)]],
    device int*        output_frontier   [[buffer(3)]],
    device atomic_int* distances         [[buffer(4)]],
    device atomic_uint* output_count     [[buffer(5)]],
    constant uint& frontier_size         [[buffer(6)]],
    constant int& iteration              [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= frontier_size) return;

    int v = input_frontier[tid];
    if (v < 0) return;

    uint start = row_offsets[v];
    uint end   = row_offsets[v + 1];
    int new_dist = iteration + 1;

    for (uint e = start; e < end; ++e) {
        int neighbor = column_indices[e];
        int old_dist = atomic_load_explicit(&distances[neighbor],
                                            memory_order_relaxed);
        if (new_dist < old_dist) {
            int prev = atomic_fetch_min_explicit(&distances[neighbor],
                                                  new_dist,
                                                  memory_order_relaxed);
            if (new_dist < prev) {
                uint idx = atomic_fetch_add_explicit(output_count, 1u,
                                                     memory_order_relaxed);
                output_frontier[idx] = neighbor;
            }
        }
    }
}

// Initialize distances: set all to INT_MAX, source to 0.
kernel void bfs_init_distances(
    device int* distances                [[buffer(0)]],
    constant uint& n_vertices            [[buffer(1)]],
    constant int& source                 [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_vertices) return;
    distances[tid] = (int(tid) == source) ? 0 : 2147483647;
}
)";
  return source;
}

}  // namespace shaders
}  // namespace metal
}  // namespace gunrock
