#include <gtest/gtest.h>
#include <gunrock/metal/context.hxx>
#include <gunrock/metal/buffer.hxx>
#include <gunrock/metal/primitives.hxx>
#include <gunrock/metal/shaders/shader_source.hxx>

#include <numeric>
#include <vector>

using namespace gunrock::metal;

class MetalPrimitivesTest : public ::testing::Test {
 protected:
  std::unique_ptr<context_t> ctx;

  void SetUp() override {
    ctx = std::make_unique<context_t>(shaders::get_all_shaders());
  }
};

TEST_F(MetalPrimitivesTest, FillInt) {
  buffer_t<int> buf(ctx->device().raw(), 500);
  primitives::fill(*ctx, buf, 77, 500);

  for (std::size_t i = 0; i < 500; ++i)
    EXPECT_EQ(buf[i], 77);
}

TEST_F(MetalPrimitivesTest, FillUint) {
  buffer_t<uint32_t> buf(ctx->device().raw(), 300);
  primitives::fill(*ctx, buf, 123u, 300);

  for (std::size_t i = 0; i < 300; ++i)
    EXPECT_EQ(buf[i], 123u);
}

TEST_F(MetalPrimitivesTest, FillFloat) {
  buffer_t<float> buf(ctx->device().raw(), 200);
  primitives::fill(*ctx, buf, 2.5f, 200);

  for (std::size_t i = 0; i < 200; ++i)
    EXPECT_FLOAT_EQ(buf[i], 2.5f);
}

TEST_F(MetalPrimitivesTest, ExclusiveScanSmall) {
  std::vector<uint32_t> input_data = {1, 2, 3, 4, 5};
  buffer_t<uint32_t> input(ctx->device().raw(), input_data.size());
  buffer_t<uint32_t> output(ctx->device().raw(), input_data.size());
  input.copy_from(input_data);

  primitives::exclusive_scan(*ctx, input, output, input_data.size());

  // Expected: [0, 1, 3, 6, 10]
  std::vector<uint32_t> expected = {0, 1, 3, 6, 10};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(output[i], expected[i]) << "Mismatch at index " << i;
}

TEST_F(MetalPrimitivesTest, ExclusiveScanOnes) {
  const std::size_t N = 256;
  buffer_t<uint32_t> input(ctx->device().raw(), N);
  buffer_t<uint32_t> output(ctx->device().raw(), N);
  for (std::size_t i = 0; i < N; ++i)
    input[i] = 1;

  primitives::exclusive_scan(*ctx, input, output, N);

  for (std::size_t i = 0; i < N; ++i)
    EXPECT_EQ(output[i], (uint32_t)i) << "Mismatch at index " << i;
}

TEST_F(MetalPrimitivesTest, ExclusiveScanMultiBlock) {
  const std::size_t N = 512;
  buffer_t<uint32_t> input(ctx->device().raw(), N);
  buffer_t<uint32_t> output(ctx->device().raw(), N);

  for (std::size_t i = 0; i < N; ++i)
    input[i] = 1;

  primitives::exclusive_scan(*ctx, input, output, N);

  for (std::size_t i = 0; i < N; ++i)
    EXPECT_EQ(output[i], (uint32_t)i) << "Mismatch at index " << i;
}

TEST_F(MetalPrimitivesTest, ReduceSumUint) {
  const std::size_t N = 256;
  buffer_t<uint32_t> input(ctx->device().raw(), N);
  for (std::size_t i = 0; i < N; ++i)
    input[i] = 1;

  uint32_t result = primitives::reduce_sum(*ctx, input, N);
  EXPECT_EQ(result, N);
}

TEST_F(MetalPrimitivesTest, ReduceSumInt) {
  const std::size_t N = 100;
  buffer_t<int32_t> input(ctx->device().raw(), N);
  for (std::size_t i = 0; i < N; ++i)
    input[i] = static_cast<int32_t>(i + 1);

  int32_t result = primitives::reduce_sum(*ctx, input, N);
  EXPECT_EQ(result, (int32_t)(N * (N + 1) / 2));
}

TEST_F(MetalPrimitivesTest, CompactRemoveNegatives) {
  std::vector<int> input_data = {1, -1, 3, -1, 5, 6, -1, 8};
  buffer_t<int> input(ctx->device().raw(), input_data.size());
  buffer_t<int> output(ctx->device().raw(), input_data.size());
  input.copy_from(input_data);

  std::size_t valid_count =
      primitives::compact(*ctx, input, output, input_data.size());

  EXPECT_EQ(valid_count, 5u);

  std::vector<int> expected = {1, 3, 5, 6, 8};
  for (std::size_t i = 0; i < valid_count; ++i)
    EXPECT_EQ(output[i], expected[i]) << "Mismatch at index " << i;
}

TEST_F(MetalPrimitivesTest, CompactAllValid) {
  std::vector<int> input_data = {10, 20, 30};
  buffer_t<int> input(ctx->device().raw(), input_data.size());
  buffer_t<int> output(ctx->device().raw(), input_data.size());
  input.copy_from(input_data);

  std::size_t valid_count =
      primitives::compact(*ctx, input, output, input_data.size());

  EXPECT_EQ(valid_count, 3u);
  for (std::size_t i = 0; i < valid_count; ++i)
    EXPECT_EQ(output[i], input_data[i]);
}

TEST_F(MetalPrimitivesTest, CompactAllInvalid) {
  std::vector<int> input_data = {-1, -1, -1};
  buffer_t<int> input(ctx->device().raw(), input_data.size());
  buffer_t<int> output(ctx->device().raw(), input_data.size());
  input.copy_from(input_data);

  std::size_t valid_count =
      primitives::compact(*ctx, input, output, input_data.size());

  EXPECT_EQ(valid_count, 0u);
}
