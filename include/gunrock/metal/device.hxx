#pragma once

#include <Metal/Metal.hpp>
#include <string>
#include <iostream>
#include <stdexcept>

namespace gunrock {
namespace metal {

using device_id_t = int;

struct device_properties_t {
  std::string name;
  std::size_t max_buffer_length;
  std::size_t max_threadgroup_memory_length;
  std::size_t max_threads_per_threadgroup;
  bool has_unified_memory;
  std::size_t recommended_max_working_set_size;
};

class device_t {
  MTL::Device* _device = nullptr;
  device_properties_t _props;

  void query_properties() {
    if (!_device)
      return;
    _props.name = _device->name()->utf8String();
    _props.max_buffer_length = _device->maxBufferLength();
    _props.max_threadgroup_memory_length =
        _device->maxThreadgroupMemoryLength();
    _props.max_threads_per_threadgroup =
        _device->maxThreadsPerThreadgroup().width;
    _props.has_unified_memory = _device->hasUnifiedMemory();
    _props.recommended_max_working_set_size =
        _device->recommendedMaxWorkingSetSize();
  }

 public:
  device_t() = default;

  explicit device_t(MTL::Device* dev) : _device(dev) {
    if (!_device)
      throw std::runtime_error("Metal device is null");
    _device->retain();
    query_properties();
  }

  ~device_t() {
    if (_device)
      _device->release();
  }

  device_t(const device_t& other) : _device(other._device), _props(other._props) {
    if (_device)
      _device->retain();
  }

  device_t& operator=(const device_t& other) {
    if (this != &other) {
      if (_device)
        _device->release();
      _device = other._device;
      _props = other._props;
      if (_device)
        _device->retain();
    }
    return *this;
  }

  device_t(device_t&& other) noexcept
      : _device(other._device), _props(std::move(other._props)) {
    other._device = nullptr;
  }

  device_t& operator=(device_t&& other) noexcept {
    if (this != &other) {
      if (_device)
        _device->release();
      _device = other._device;
      _props = std::move(other._props);
      other._device = nullptr;
    }
    return *this;
  }

  static device_t create_default() {
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    if (!dev)
      throw std::runtime_error("No Metal device found on this system");
    device_t d(dev);
    dev->release();
    return d;
  }

  MTL::Device* raw() const { return _device; }
  const device_properties_t& properties() const { return _props; }
  const std::string& name() const { return _props.name; }
  std::size_t max_buffer_length() const { return _props.max_buffer_length; }
  std::size_t max_threads_per_threadgroup() const {
    return _props.max_threads_per_threadgroup;
  }
  bool has_unified_memory() const { return _props.has_unified_memory; }

  void print_properties() const {
    std::cout << "Metal Device Properties:" << std::endl;
    std::cout << "  Name:                      " << _props.name << std::endl;
    std::cout << "  Unified Memory:            "
              << (_props.has_unified_memory ? "Yes" : "No") << std::endl;
    std::cout << "  Max Buffer Length:         "
              << (_props.max_buffer_length / (1024 * 1024)) << " MB"
              << std::endl;
    std::cout << "  Max Threadgroup Memory:    "
              << _props.max_threadgroup_memory_length << " bytes" << std::endl;
    std::cout << "  Max Threads/Threadgroup:   "
              << _props.max_threads_per_threadgroup << std::endl;
    std::cout << "  Recommended Working Set:   "
              << (_props.recommended_max_working_set_size / (1024 * 1024))
              << " MB" << std::endl;
  }
};

}  // namespace metal
}  // namespace gunrock
