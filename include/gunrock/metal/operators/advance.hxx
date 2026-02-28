#pragma once

#include <gunrock/metal/context.hxx>
#include <gunrock/metal/buffer.hxx>
#include <gunrock/metal/graph.hxx>
#include <gunrock/metal/primitives.hxx>

namespace gunrock {
namespace metal {
namespace operators {
namespace advance {

// Thread-mapped advance: expand the input frontier by visiting all neighbors.
// Returns the size of the output frontier.
template <typename graph_t>
std::size_t execute(context_t& ctx,
                    graph_t& G,
                    buffer_t<typename graph_t::vertex_type>& input_frontier,
                    std::size_t input_size,
                    buffer_t<typename graph_t::vertex_type>& output_frontier,
                    buffer_t<typename graph_t::edge_type>& segment_offsets) {
  if (input_size == 0)
    return 0;

  buffer_t<uint32_t> frontier_size_buf(ctx.device().raw(), 1);
  frontier_size_buf[0] = static_cast<uint32_t>(input_size);

  // Step 1: compute per-vertex output sizes
  buffer_t<uint32_t> output_sizes(ctx.device().raw(), input_size);

  ctx.dispatch_1d(
      "compute_output_offsets", input_size,
      [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(G.row_offsets().raw(), 0, 0);
        enc->setBuffer(input_frontier.raw(), 0, 1);
        enc->setBuffer(output_sizes.raw(), 0, 2);
        enc->setBuffer(frontier_size_buf.raw(), 0, 3);
      });

  // Step 2: exclusive scan of output sizes -> segment offsets
  if (segment_offsets.size() < input_size)
    segment_offsets.resize(ctx.device().raw(), input_size);

  primitives::exclusive_scan(ctx, output_sizes, segment_offsets, input_size);

  // Compute total output size
  uint32_t last_size = output_sizes[input_size - 1];
  uint32_t last_offset = segment_offsets[input_size - 1];
  std::size_t total_output = last_offset + last_size;

  if (total_output == 0)
    return 0;

  // Step 3: ensure output frontier is large enough
  if (output_frontier.size() < total_output)
    output_frontier.resize(ctx.device().raw(), total_output);

  // Step 4: dispatch thread-mapped advance
  ctx.dispatch_1d(
      "advance_thread_mapped", input_size,
      [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(G.row_offsets().raw(), 0, 0);
        enc->setBuffer(G.column_indices().raw(), 0, 1);
        enc->setBuffer(input_frontier.raw(), 0, 2);
        enc->setBuffer(output_frontier.raw(), 0, 3);
        enc->setBuffer(segment_offsets.raw(), 0, 4);
        enc->setBuffer(frontier_size_buf.raw(), 0, 5);
      });

  return total_output;
}

}  // namespace advance
}  // namespace operators
}  // namespace metal
}  // namespace gunrock
