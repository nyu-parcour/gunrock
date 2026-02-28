#pragma once

#include <gunrock/metal/buffer.hxx>
#include <gunrock/metal/device.hxx>

#include <vector>
#include <stdexcept>

namespace gunrock {
namespace metal {

template <typename vertex_t = int,
          typename edge_t = uint32_t,
          typename weight_t = float>
class graph_t {
 public:
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using weight_type = weight_t;

 private:
  buffer_t<edge_t> _row_offsets;
  buffer_t<vertex_t> _column_indices;
  buffer_t<weight_t> _edge_weights;

  std::size_t _n_vertices = 0;
  std::size_t _n_edges = 0;
  bool _is_directed = true;
  bool _has_weights = false;

 public:
  graph_t() = default;

  graph_t(graph_t&&) = default;
  graph_t& operator=(graph_t&&) = default;

  static graph_t from_csr(MTL::Device* device,
                          const std::vector<edge_t>& offsets,
                          const std::vector<vertex_t>& indices,
                          std::size_t n_vertices,
                          std::size_t n_edges,
                          bool directed = true) {
    graph_t g;
    g._n_vertices = n_vertices;
    g._n_edges = n_edges;
    g._is_directed = directed;
    g._has_weights = false;

    g._row_offsets = buffer_t<edge_t>(device, offsets.size());
    g._row_offsets.copy_from(offsets);

    g._column_indices = buffer_t<vertex_t>(device, indices.size());
    g._column_indices.copy_from(indices);

    return g;
  }

  static graph_t from_csr(MTL::Device* device,
                          const std::vector<edge_t>& offsets,
                          const std::vector<vertex_t>& indices,
                          const std::vector<weight_t>& weights,
                          std::size_t n_vertices,
                          std::size_t n_edges,
                          bool directed = true) {
    graph_t g = from_csr(device, offsets, indices, n_vertices, n_edges, directed);
    g._has_weights = true;
    g._edge_weights = buffer_t<weight_t>(device, weights.size());
    g._edge_weights.copy_from(weights);
    return g;
  }

  std::size_t get_number_of_vertices() const { return _n_vertices; }
  std::size_t get_number_of_edges() const { return _n_edges; }
  bool is_directed() const { return _is_directed; }
  bool has_weights() const { return _has_weights; }

  buffer_t<edge_t>& row_offsets() { return _row_offsets; }
  const buffer_t<edge_t>& row_offsets() const { return _row_offsets; }

  buffer_t<vertex_t>& column_indices() { return _column_indices; }
  const buffer_t<vertex_t>& column_indices() const { return _column_indices; }

  buffer_t<weight_t>& edge_weights() { return _edge_weights; }
  const buffer_t<weight_t>& edge_weights() const { return _edge_weights; }

  edge_t get_number_of_neighbors(vertex_t v) const {
    return _row_offsets.data()[v + 1] - _row_offsets.data()[v];
  }

  edge_t get_starting_edge(vertex_t v) const {
    return _row_offsets.data()[v];
  }

  vertex_t get_destination_vertex(edge_t e) const {
    return _column_indices.data()[e];
  }
};

}  // namespace metal
}  // namespace gunrock
