#include <gtest/gtest.h>
#include <gunrock/metal/context.hxx>
#include <gunrock/metal/buffer.hxx>
#include <gunrock/metal/shaders/shader_source.hxx>

using namespace gunrock::metal;

class MetalComputeTest : public ::testing::Test {
 protected:
  std::unique_ptr<context_t> ctx;

  void SetUp() override {
    ctx = std::make_unique<context_t>(shaders::get_all_shaders());
  }
};

TEST_F(MetalComputeTest, CreateContext) {
  EXPECT_NE(ctx->queue(), nullptr);
  EXPECT_NE(ctx->library(), nullptr);
}

TEST_F(MetalComputeTest, GetPipeline) {
  auto pso = ctx->get_pipeline("fill_int");
  EXPECT_NE(pso, nullptr);
}

TEST_F(MetalComputeTest, FillInt) {
  const std::size_t N = 256;
  buffer_t<int> buf(ctx->device().raw(), N);
  buf.fill(0);

  buffer_t<uint32_t> count_buf(ctx->device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(N);
  buffer_t<int> val_buf(ctx->device().raw(), 1);
  val_buf[0] = 42;

  ctx->dispatch_1d("fill_int", N, [&](MTL::ComputeCommandEncoder* enc) {
    enc->setBuffer(buf.raw(), 0, 0);
    enc->setBuffer(count_buf.raw(), 0, 1);
    enc->setBuffer(val_buf.raw(), 0, 2);
  });

  for (std::size_t i = 0; i < N; ++i)
    EXPECT_EQ(buf[i], 42) << "Mismatch at index " << i;
}

TEST_F(MetalComputeTest, FillFloat) {
  const std::size_t N = 128;
  buffer_t<float> buf(ctx->device().raw(), N);

  buffer_t<uint32_t> count_buf(ctx->device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(N);
  buffer_t<float> val_buf(ctx->device().raw(), 1);
  val_buf[0] = 3.14f;

  ctx->dispatch_1d("fill_float", N, [&](MTL::ComputeCommandEncoder* enc) {
    enc->setBuffer(buf.raw(), 0, 0);
    enc->setBuffer(count_buf.raw(), 0, 1);
    enc->setBuffer(val_buf.raw(), 0, 2);
  });

  for (std::size_t i = 0; i < N; ++i)
    EXPECT_FLOAT_EQ(buf[i], 3.14f) << "Mismatch at index " << i;
}

TEST_F(MetalComputeTest, FillLargeBuffer) {
  const std::size_t N = 10000;
  buffer_t<int> buf(ctx->device().raw(), N);

  buffer_t<uint32_t> count_buf(ctx->device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(N);
  buffer_t<int> val_buf(ctx->device().raw(), 1);
  val_buf[0] = -1;

  ctx->dispatch_1d("fill_int", N, [&](MTL::ComputeCommandEncoder* enc) {
    enc->setBuffer(buf.raw(), 0, 0);
    enc->setBuffer(count_buf.raw(), 0, 1);
    enc->setBuffer(val_buf.raw(), 0, 2);
  });

  for (std::size_t i = 0; i < N; ++i)
    EXPECT_EQ(buf[i], -1) << "Mismatch at index " << i;
}
