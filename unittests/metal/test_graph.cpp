#include <gtest/gtest.h>
#include <gunrock/metal/device.hxx>
#include <gunrock/metal/graph.hxx>

#include <vector>

using namespace gunrock::metal;

class MetalGraphTest : public ::testing::Test {
 protected:
  device_t dev;

  // Small test graph (directed):
  //   0 -> 1, 2
  //   1 -> 2, 3
  //   2 -> 3
  //   3 -> (none)
  //
  // CSR: offsets = [0, 2, 4, 5, 5]
  //      indices = [1, 2, 2, 3, 3]
  std::vector<uint32_t> offsets = {0, 2, 4, 5, 5};
  std::vector<int> indices = {1, 2, 2, 3, 3};
  std::size_t n_vertices = 4;
  std::size_t n_edges = 5;

  void SetUp() override { dev = device_t::create_default(); }
};

TEST_F(MetalGraphTest, BuildUnweighted) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      dev.raw(), offsets, indices, n_vertices, n_edges);

  EXPECT_EQ(g.get_number_of_vertices(), n_vertices);
  EXPECT_EQ(g.get_number_of_edges(), n_edges);
  EXPECT_TRUE(g.is_directed());
  EXPECT_FALSE(g.has_weights());
}

TEST_F(MetalGraphTest, BuildWeighted) {
  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto g = graph_t<int, uint32_t, float>::from_csr(
      dev.raw(), offsets, indices, weights, n_vertices, n_edges);

  EXPECT_TRUE(g.has_weights());
  EXPECT_EQ(g.edge_weights().size(), n_edges);
  EXPECT_FLOAT_EQ(g.edge_weights()[0], 1.0f);
  EXPECT_FLOAT_EQ(g.edge_weights()[4], 5.0f);
}

TEST_F(MetalGraphTest, RowOffsetsAccessible) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      dev.raw(), offsets, indices, n_vertices, n_edges);

  for (std::size_t i = 0; i < offsets.size(); ++i)
    EXPECT_EQ(g.row_offsets()[i], offsets[i]);
}

TEST_F(MetalGraphTest, ColumnIndicesAccessible) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      dev.raw(), offsets, indices, n_vertices, n_edges);

  for (std::size_t i = 0; i < indices.size(); ++i)
    EXPECT_EQ(g.column_indices()[i], indices[i]);
}

TEST_F(MetalGraphTest, NeighborCounts) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      dev.raw(), offsets, indices, n_vertices, n_edges);

  EXPECT_EQ(g.get_number_of_neighbors(0), 2u);
  EXPECT_EQ(g.get_number_of_neighbors(1), 2u);
  EXPECT_EQ(g.get_number_of_neighbors(2), 1u);
  EXPECT_EQ(g.get_number_of_neighbors(3), 0u);
}

TEST_F(MetalGraphTest, StartingEdges) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      dev.raw(), offsets, indices, n_vertices, n_edges);

  EXPECT_EQ(g.get_starting_edge(0), 0u);
  EXPECT_EQ(g.get_starting_edge(1), 2u);
  EXPECT_EQ(g.get_starting_edge(2), 4u);
  EXPECT_EQ(g.get_starting_edge(3), 5u);
}

TEST_F(MetalGraphTest, DestinationVertices) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      dev.raw(), offsets, indices, n_vertices, n_edges);

  EXPECT_EQ(g.get_destination_vertex(0), 1);
  EXPECT_EQ(g.get_destination_vertex(1), 2);
  EXPECT_EQ(g.get_destination_vertex(2), 2);
  EXPECT_EQ(g.get_destination_vertex(3), 3);
  EXPECT_EQ(g.get_destination_vertex(4), 3);
}
