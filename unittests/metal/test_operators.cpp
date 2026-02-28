#include <gtest/gtest.h>
#include <gunrock/metal/context.hxx>
#include <gunrock/metal/graph.hxx>
#include <gunrock/metal/operators/advance.hxx>
#include <gunrock/metal/operators/filter.hxx>
#include <gunrock/metal/shaders/shader_source.hxx>

#include <algorithm>
#include <set>
#include <vector>

using namespace gunrock::metal;

class MetalOperatorsTest : public ::testing::Test {
 protected:
  std::unique_ptr<context_t> ctx;

  // Test graph:
  //   0 -> 1, 2
  //   1 -> 2, 3
  //   2 -> 3
  //   3 -> (none)
  std::vector<uint32_t> offsets = {0, 2, 4, 5, 5};
  std::vector<int> indices = {1, 2, 2, 3, 3};
  std::size_t n_vertices = 4;
  std::size_t n_edges = 5;

  void SetUp() override {
    ctx = std::make_unique<context_t>(shaders::get_all_shaders());
  }
};

TEST_F(MetalOperatorsTest, AdvanceFromVertex0) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, n_vertices, n_edges);

  buffer_t<int> input(ctx->device().raw(), 1);
  input[0] = 0;

  buffer_t<int> output(ctx->device().raw(), n_edges);
  buffer_t<uint32_t> segments(ctx->device().raw(), n_vertices);

  std::size_t out_size =
      operators::advance::execute(*ctx, g, input, 1, output, segments);

  EXPECT_EQ(out_size, 2u);
  std::set<int> neighbors;
  for (std::size_t i = 0; i < out_size; ++i)
    neighbors.insert(output[i]);

  EXPECT_TRUE(neighbors.count(1));
  EXPECT_TRUE(neighbors.count(2));
}

TEST_F(MetalOperatorsTest, AdvanceFromVertex1) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, n_vertices, n_edges);

  buffer_t<int> input(ctx->device().raw(), 1);
  input[0] = 1;

  buffer_t<int> output(ctx->device().raw(), n_edges);
  buffer_t<uint32_t> segments(ctx->device().raw(), n_vertices);

  std::size_t out_size =
      operators::advance::execute(*ctx, g, input, 1, output, segments);

  EXPECT_EQ(out_size, 2u);
  std::set<int> neighbors;
  for (std::size_t i = 0; i < out_size; ++i)
    neighbors.insert(output[i]);

  EXPECT_TRUE(neighbors.count(2));
  EXPECT_TRUE(neighbors.count(3));
}

TEST_F(MetalOperatorsTest, AdvanceFromMultipleVertices) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, n_vertices, n_edges);

  buffer_t<int> input(ctx->device().raw(), 2);
  input[0] = 0;
  input[1] = 1;

  buffer_t<int> output(ctx->device().raw(), n_edges);
  buffer_t<uint32_t> segments(ctx->device().raw(), n_vertices);

  std::size_t out_size =
      operators::advance::execute(*ctx, g, input, 2, output, segments);

  EXPECT_EQ(out_size, 4u);
}

TEST_F(MetalOperatorsTest, AdvanceFromLeafVertex) {
  auto g = graph_t<int, uint32_t, float>::from_csr(
      ctx->device().raw(), offsets, indices, n_vertices, n_edges);

  buffer_t<int> input(ctx->device().raw(), 1);
  input[0] = 3;

  buffer_t<int> output(ctx->device().raw(), n_edges);
  buffer_t<uint32_t> segments(ctx->device().raw(), n_vertices);

  std::size_t out_size =
      operators::advance::execute(*ctx, g, input, 1, output, segments);

  EXPECT_EQ(out_size, 0u);
}

TEST_F(MetalOperatorsTest, FilterRemovesInvalid) {
  std::vector<int> data = {5, -1, 3, -1, 7, -1, 2};
  buffer_t<int> input(ctx->device().raw(), data.size());
  buffer_t<int> output(ctx->device().raw(), data.size());
  input.copy_from(data);

  std::size_t count =
      operators::filter::execute(*ctx, input, output, data.size());

  EXPECT_EQ(count, 4u);
  std::vector<int> expected = {5, 3, 7, 2};
  for (std::size_t i = 0; i < count; ++i)
    EXPECT_EQ(output[i], expected[i]);
}
