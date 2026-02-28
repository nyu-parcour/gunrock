#pragma once

#include <gunrock/metal/context.hxx>
#include <gunrock/metal/buffer.hxx>
#include <gunrock/metal/graph.hxx>
#include <gunrock/metal/primitives.hxx>
#include <gunrock/metal/operators/filter.hxx>

#include <limits>
#include <vector>

namespace gunrock {
namespace metal {
namespace bfs {

template <typename vertex_t = int,
          typename edge_t = uint32_t,
          typename weight_t = float>
struct result_t {
  std::vector<vertex_t> distances;
  int iterations = 0;
  float elapsed_ms = 0.0f;
};

template <typename graph_t>
result_t<typename graph_t::vertex_type,
         typename graph_t::edge_type,
         typename graph_t::weight_type>
run(context_t& ctx,
    graph_t& G,
    typename graph_t::vertex_type source) {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  const std::size_t n_vertices = G.get_number_of_vertices();
  const std::size_t n_edges = G.get_number_of_edges();

  // Allocate distances buffer and initialize via GPU kernel
  buffer_t<vertex_t> distances(ctx.device().raw(), n_vertices);
  {
    buffer_t<uint32_t> nv_buf(ctx.device().raw(), 1);
    buffer_t<vertex_t> src_buf(ctx.device().raw(), 1);
    nv_buf[0] = static_cast<uint32_t>(n_vertices);
    src_buf[0] = source;

    ctx.dispatch_1d(
        "bfs_init_distances", n_vertices,
        [&](MTL::ComputeCommandEncoder* enc) {
          enc->setBuffer(distances.raw(), 0, 0);
          enc->setBuffer(nv_buf.raw(), 0, 1);
          enc->setBuffer(src_buf.raw(), 0, 2);
        });
  }

  // Allocate frontier buffers (double-buffered)
  std::size_t frontier_capacity = std::max(n_edges, n_vertices);
  buffer_t<vertex_t> frontier_a(ctx.device().raw(), frontier_capacity);
  buffer_t<vertex_t> frontier_b(ctx.device().raw(), frontier_capacity);
  buffer_t<uint32_t> output_count(ctx.device().raw(), 1);

  // Initialize input frontier = {source}
  frontier_a[0] = source;
  std::size_t current_size = 1;

  buffer_t<vertex_t>* input = &frontier_a;
  buffer_t<vertex_t>* output = &frontier_b;

  int iteration = 0;
  auto& timer = ctx.timer();
  timer.begin();

  while (current_size > 0) {
    // Reset output counter
    output_count[0] = 0;

    buffer_t<uint32_t> frontier_size_buf(ctx.device().raw(), 1);
    frontier_size_buf[0] = static_cast<uint32_t>(current_size);

    buffer_t<int32_t> iter_buf(ctx.device().raw(), 1);
    iter_buf[0] = iteration;

    // Dispatch BFS advance kernel
    ctx.dispatch_1d(
        "bfs_advance", current_size,
        [&](MTL::ComputeCommandEncoder* enc) {
          enc->setBuffer(G.row_offsets().raw(), 0, 0);
          enc->setBuffer(G.column_indices().raw(), 0, 1);
          enc->setBuffer(input->raw(), 0, 2);
          enc->setBuffer(output->raw(), 0, 3);
          enc->setBuffer(distances.raw(), 0, 4);
          enc->setBuffer(output_count.raw(), 0, 5);
          enc->setBuffer(frontier_size_buf.raw(), 0, 6);
          enc->setBuffer(iter_buf.raw(), 0, 7);
        });

    // Read back output frontier size
    current_size = output_count[0];

    // Swap frontiers
    std::swap(input, output);
    ++iteration;
  }

  float elapsed = timer.end();

  // Copy results to host
  result_t<vertex_t, edge_t, weight_t> result;
  result.iterations = iteration;
  result.elapsed_ms = elapsed;
  distances.copy_to(result.distances);

  return result;
}

}  // namespace bfs
}  // namespace metal
}  // namespace gunrock
