#include <gtest/gtest.h>
#include <gunrock/metal/context.hxx>
#include <gunrock/metal/graph.hxx>
#include <gunrock/metal/algorithms/bfs.hxx>
#include <gunrock/metal/shaders/shader_source.hxx>

#include <limits>
#include <vector>

using namespace gunrock::metal;

class MetalBFSTest : public ::testing::Test {
 protected:
  std::unique_ptr<context_t> ctx;

  void SetUp() override {
    ctx = std::make_unique<context_t>(shaders::get_all_shaders());
  }
};

// Linear graph: 0 -> 1 -> 2 -> 3 -> 4
TEST_F(MetalBFSTest, LinearGraph) {
  std::vector<uint32_t> offsets = {0, 1, 2, 3, 4, 4};
  std::vector<int> indices = {1, 2, 3, 4};
  std::size_t nv = 5, ne = 4;

  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, nv, ne);

  auto result = bfs::run(*ctx, g, 0);

  EXPECT_EQ(result.distances.size(), nv);
  EXPECT_EQ(result.distances[0], 0);
  EXPECT_EQ(result.distances[1], 1);
  EXPECT_EQ(result.distances[2], 2);
  EXPECT_EQ(result.distances[3], 3);
  EXPECT_EQ(result.distances[4], 4);
  EXPECT_GT(result.iterations, 0);
}

// Star graph: 0 -> 1, 2, 3, 4
TEST_F(MetalBFSTest, StarGraph) {
  std::vector<uint32_t> offsets = {0, 4, 4, 4, 4, 4};
  std::vector<int> indices = {1, 2, 3, 4};
  std::size_t nv = 5, ne = 4;

  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, nv, ne);

  auto result = bfs::run(*ctx, g, 0);

  EXPECT_EQ(result.distances[0], 0);
  EXPECT_EQ(result.distances[1], 1);
  EXPECT_EQ(result.distances[2], 1);
  EXPECT_EQ(result.distances[3], 1);
  EXPECT_EQ(result.distances[4], 1);
  EXPECT_GT(result.iterations, 0);
}

// Small DAG:
//   0 -> 1, 2
//   1 -> 3
//   2 -> 3
//   3 -> (none)
TEST_F(MetalBFSTest, SmallDAG) {
  std::vector<uint32_t> offsets = {0, 2, 3, 4, 4};
  std::vector<int> indices = {1, 2, 3, 3};
  std::size_t nv = 4, ne = 4;

  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, nv, ne);

  auto result = bfs::run(*ctx, g, 0);

  EXPECT_EQ(result.distances[0], 0);
  EXPECT_EQ(result.distances[1], 1);
  EXPECT_EQ(result.distances[2], 1);
  EXPECT_EQ(result.distances[3], 2);
}

// Undirected triangle: 0-1-2-0 (stored as directed edges both ways)
TEST_F(MetalBFSTest, UndirectedTriangle) {
  //   0 -> 1, 2
  //   1 -> 0, 2
  //   2 -> 0, 1
  std::vector<uint32_t> offsets = {0, 2, 4, 6};
  std::vector<int> indices = {1, 2, 0, 2, 0, 1};
  std::size_t nv = 3, ne = 6;

  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, nv, ne, false);

  auto result = bfs::run(*ctx, g, 0);

  EXPECT_EQ(result.distances[0], 0);
  EXPECT_EQ(result.distances[1], 1);
  EXPECT_EQ(result.distances[2], 1);
}

// Disconnected: vertices 3 and 4 unreachable from source 0
TEST_F(MetalBFSTest, DisconnectedGraph) {
  // 0 -> 1, 2
  // 1 -> (none)
  // 2 -> (none)
  // 3 -> 4
  // 4 -> (none)
  std::vector<uint32_t> offsets = {0, 2, 2, 2, 3, 3};
  std::vector<int> indices = {1, 2, 4};
  std::size_t nv = 5, ne = 3;

  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, nv, ne);

  auto result = bfs::run(*ctx, g, 0);

  EXPECT_EQ(result.distances[0], 0);
  EXPECT_EQ(result.distances[1], 1);
  EXPECT_EQ(result.distances[2], 1);
  EXPECT_EQ(result.distances[3], std::numeric_limits<int>::max());
  EXPECT_EQ(result.distances[4], std::numeric_limits<int>::max());
}

// BFS from non-zero source
TEST_F(MetalBFSTest, NonZeroSource) {
  // 0 -> 1
  // 1 -> 2
  // 2 -> 0
  std::vector<uint32_t> offsets = {0, 1, 2, 3};
  std::vector<int> indices = {1, 2, 0};
  std::size_t nv = 3, ne = 3;

  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, nv, ne);

  auto result = bfs::run(*ctx, g, 1);

  EXPECT_EQ(result.distances[0], 2);
  EXPECT_EQ(result.distances[1], 0);
  EXPECT_EQ(result.distances[2], 1);
}

// Single vertex
TEST_F(MetalBFSTest, SingleVertex) {
  std::vector<uint32_t> offsets = {0, 0};
  std::vector<int> indices = {};
  std::size_t nv = 1, ne = 0;

  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, nv, ne);

  auto result = bfs::run(*ctx, g, 0);

  EXPECT_EQ(result.distances[0], 0);
}

TEST_F(MetalBFSTest, ElapsedTimePositive) {
  std::vector<uint32_t> offsets = {0, 2, 3, 4, 4};
  std::vector<int> indices = {1, 2, 3, 3};
  std::size_t nv = 4, ne = 4;

  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, nv, ne);

  auto result = bfs::run(*ctx, g, 0);
  EXPECT_GE(result.elapsed_ms, 0.0f);
}
