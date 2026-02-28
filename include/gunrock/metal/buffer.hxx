#pragma once

#include <Metal/Metal.hpp>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <limits>

namespace gunrock {
namespace metal {

template <typename T>
class buffer_t {
  MTL::Buffer* _buffer = nullptr;
  std::size_t _count = 0;

 public:
  using value_type = T;

  buffer_t() = default;

  buffer_t(MTL::Device* device, std::size_t count)
      : _count(count) {
    if (count == 0)
      return;
    _buffer = device->newBuffer(count * sizeof(T),
                                MTL::ResourceStorageModeShared);
    if (!_buffer)
      throw std::runtime_error("Failed to allocate Metal buffer");
  }

  ~buffer_t() {
    if (_buffer)
      _buffer->release();
  }

  buffer_t(const buffer_t&) = delete;
  buffer_t& operator=(const buffer_t&) = delete;

  buffer_t(buffer_t&& other) noexcept
      : _buffer(other._buffer), _count(other._count) {
    other._buffer = nullptr;
    other._count = 0;
  }

  buffer_t& operator=(buffer_t&& other) noexcept {
    if (this != &other) {
      if (_buffer)
        _buffer->release();
      _buffer = other._buffer;
      _count = other._count;
      other._buffer = nullptr;
      other._count = 0;
    }
    return *this;
  }

  T* data() {
    return _buffer ? static_cast<T*>(_buffer->contents()) : nullptr;
  }

  const T* data() const {
    return _buffer ? static_cast<const T*>(_buffer->contents()) : nullptr;
  }

  std::size_t size() const { return _count; }
  std::size_t byte_size() const { return _count * sizeof(T); }
  bool empty() const { return _count == 0; }

  MTL::Buffer* raw() const { return _buffer; }

  T& operator[](std::size_t idx) { return data()[idx]; }
  const T& operator[](std::size_t idx) const { return data()[idx]; }

  T* begin() { return data(); }
  T* end() { return data() + _count; }
  const T* begin() const { return data(); }
  const T* end() const { return data() + _count; }

  void fill(T value) {
    T* ptr = data();
    for (std::size_t i = 0; i < _count; ++i)
      ptr[i] = value;
  }

  void copy_from(const T* src, std::size_t count) {
    if (count > _count)
      throw std::runtime_error("copy_from: source exceeds buffer capacity");
    std::memcpy(data(), src, count * sizeof(T));
  }

  void copy_from(const std::vector<T>& src) {
    copy_from(src.data(), src.size());
  }

  void copy_to(T* dst, std::size_t count) const {
    if (count > _count)
      count = _count;
    std::memcpy(dst, data(), count * sizeof(T));
  }

  void copy_to(std::vector<T>& dst) const {
    dst.resize(_count);
    copy_to(dst.data(), _count);
  }

  void resize(MTL::Device* device, std::size_t new_count) {
    if (new_count == _count)
      return;
    MTL::Buffer* new_buf = device->newBuffer(
        new_count * sizeof(T), MTL::ResourceStorageModeShared);
    if (!new_buf)
      throw std::runtime_error("Failed to resize Metal buffer");
    if (_buffer && _count > 0) {
      std::size_t copy_count = std::min(_count, new_count);
      std::memcpy(new_buf->contents(), _buffer->contents(),
                  copy_count * sizeof(T));
      _buffer->release();
    }
    _buffer = new_buf;
    _count = new_count;
  }
};

}  // namespace metal
}  // namespace gunrock
