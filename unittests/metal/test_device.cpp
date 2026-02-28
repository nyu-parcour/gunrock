#include <gtest/gtest.h>
#include <gunrock/metal/device.hxx>

using namespace gunrock::metal;

TEST(MetalDevice, CreateDefault) {
  device_t dev = device_t::create_default();
  EXPECT_NE(dev.raw(), nullptr);
}

TEST(MetalDevice, NameNonEmpty) {
  device_t dev = device_t::create_default();
  EXPECT_FALSE(dev.name().empty());
}

TEST(MetalDevice, UnifiedMemory) {
  device_t dev = device_t::create_default();
  EXPECT_TRUE(dev.has_unified_memory());
}

TEST(MetalDevice, MaxBufferLength) {
  device_t dev = device_t::create_default();
  EXPECT_GT(dev.max_buffer_length(), 0u);
}

TEST(MetalDevice, MaxThreadsPerThreadgroup) {
  device_t dev = device_t::create_default();
  EXPECT_GT(dev.max_threads_per_threadgroup(), 0u);
}

TEST(MetalDevice, CopyConstruct) {
  device_t dev = device_t::create_default();
  device_t copy(dev);
  EXPECT_EQ(dev.name(), copy.name());
  EXPECT_EQ(dev.raw(), copy.raw());
}

TEST(MetalDevice, MoveConstruct) {
  device_t dev = device_t::create_default();
  std::string name = dev.name();
  device_t moved(std::move(dev));
  EXPECT_EQ(moved.name(), name);
}

TEST(MetalDevice, PrintProperties) {
  device_t dev = device_t::create_default();
  EXPECT_NO_THROW(dev.print_properties());
}
