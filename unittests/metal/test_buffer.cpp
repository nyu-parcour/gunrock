#include <gtest/gtest.h>
#include <gunrock/metal/device.hxx>
#include <gunrock/metal/buffer.hxx>

#include <vector>
#include <numeric>

using namespace gunrock::metal;

class MetalBufferTest : public ::testing::Test {
 protected:
  device_t dev;
  void SetUp() override { dev = device_t::create_default(); }
};

TEST_F(MetalBufferTest, AllocateAndSize) {
  buffer_t<int> buf(dev.raw(), 1024);
  EXPECT_EQ(buf.size(), 1024u);
  EXPECT_EQ(buf.byte_size(), 1024u * sizeof(int));
  EXPECT_NE(buf.data(), nullptr);
  EXPECT_NE(buf.raw(), nullptr);
}

TEST_F(MetalBufferTest, EmptyBuffer) {
  buffer_t<int> buf;
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_TRUE(buf.empty());
}

TEST_F(MetalBufferTest, WriteAndRead) {
  buffer_t<int> buf(dev.raw(), 10);
  for (int i = 0; i < 10; ++i)
    buf[i] = i * 10;

  for (int i = 0; i < 10; ++i)
    EXPECT_EQ(buf[i], i * 10);
}

TEST_F(MetalBufferTest, Fill) {
  buffer_t<int> buf(dev.raw(), 100);
  buf.fill(42);

  for (std::size_t i = 0; i < buf.size(); ++i)
    EXPECT_EQ(buf[i], 42);
}

TEST_F(MetalBufferTest, CopyFromVector) {
  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  buffer_t<float> buf(dev.raw(), host_data.size());
  buf.copy_from(host_data);

  for (std::size_t i = 0; i < host_data.size(); ++i)
    EXPECT_FLOAT_EQ(buf[i], host_data[i]);
}

TEST_F(MetalBufferTest, CopyToVector) {
  buffer_t<int> buf(dev.raw(), 5);
  for (int i = 0; i < 5; ++i)
    buf[i] = i + 100;

  std::vector<int> host_data;
  buf.copy_to(host_data);

  EXPECT_EQ(host_data.size(), 5u);
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(host_data[i], i + 100);
}

TEST_F(MetalBufferTest, MoveConstruct) {
  buffer_t<int> buf(dev.raw(), 10);
  buf[0] = 99;
  buffer_t<int> moved(std::move(buf));

  EXPECT_EQ(moved.size(), 10u);
  EXPECT_EQ(moved[0], 99);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(buf.data(), nullptr);
}

TEST_F(MetalBufferTest, Resize) {
  buffer_t<int> buf(dev.raw(), 5);
  for (int i = 0; i < 5; ++i)
    buf[i] = i;

  buf.resize(dev.raw(), 10);
  EXPECT_EQ(buf.size(), 10u);

  // First 5 elements should be preserved
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(buf[i], i);
}

TEST_F(MetalBufferTest, BeginEnd) {
  buffer_t<int> buf(dev.raw(), 5);
  std::iota(buf.begin(), buf.end(), 0);

  int sum = 0;
  for (auto it = buf.begin(); it != buf.end(); ++it)
    sum += *it;
  EXPECT_EQ(sum, 10);
}
