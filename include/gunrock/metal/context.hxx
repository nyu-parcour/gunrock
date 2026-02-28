#pragma once

#include <Metal/Metal.hpp>
#include <gunrock/metal/device.hxx>

#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace gunrock {
namespace metal {

class timer_t {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point _start;
  clock::time_point _stop;

 public:
  void begin() { _start = clock::now(); }
  float end() {
    _stop = clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(_stop - _start);
    return duration.count() / 1000.0f;
  }
};

class context_t {
  device_t _device;
  MTL::CommandQueue* _queue = nullptr;
  MTL::Library* _library = nullptr;
  std::unordered_map<std::string, MTL::ComputePipelineState*> _pipeline_cache;
  timer_t _timer;

  void compile_library(const std::string& source) {
    NS::Error* error = nullptr;
    auto src = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    _library = _device.raw()->newLibrary(src, nullptr, &error);
    if (!_library) {
      std::string msg = "Failed to compile Metal library";
      if (error)
        msg += std::string(": ") + error->localizedDescription()->utf8String();
      throw std::runtime_error(msg);
    }
  }

 public:
  context_t() = default;

  explicit context_t(const std::string& shader_source, device_id_t dev_id = 0)
      : _device(device_t::create_default()) {
    _queue = _device.raw()->newCommandQueue();
    if (!_queue)
      throw std::runtime_error("Failed to create Metal command queue");
    compile_library(shader_source);
  }

  context_t(const device_t& device, const std::string& shader_source)
      : _device(device) {
    _queue = _device.raw()->newCommandQueue();
    if (!_queue)
      throw std::runtime_error("Failed to create Metal command queue");
    compile_library(shader_source);
  }

  ~context_t() {
    for (auto& [name, pso] : _pipeline_cache)
      pso->release();
    if (_library)
      _library->release();
    if (_queue)
      _queue->release();
  }

  context_t(const context_t&) = delete;
  context_t& operator=(const context_t&) = delete;

  context_t(context_t&& other) noexcept
      : _device(std::move(other._device)),
        _queue(other._queue),
        _library(other._library),
        _pipeline_cache(std::move(other._pipeline_cache)),
        _timer(other._timer) {
    other._queue = nullptr;
    other._library = nullptr;
  }

  context_t& operator=(context_t&& other) noexcept {
    if (this != &other) {
      for (auto& [name, pso] : _pipeline_cache)
        pso->release();
      if (_library)
        _library->release();
      if (_queue)
        _queue->release();
      _device = std::move(other._device);
      _queue = other._queue;
      _library = other._library;
      _pipeline_cache = std::move(other._pipeline_cache);
      _timer = other._timer;
      other._queue = nullptr;
      other._library = nullptr;
    }
    return *this;
  }

  device_t& device() { return _device; }
  const device_t& device() const { return _device; }
  MTL::CommandQueue* queue() const { return _queue; }
  MTL::Library* library() const { return _library; }
  timer_t& timer() { return _timer; }

  MTL::ComputePipelineState* get_pipeline(const std::string& fn_name) {
    auto it = _pipeline_cache.find(fn_name);
    if (it != _pipeline_cache.end())
      return it->second;

    auto fn_str =
        NS::String::string(fn_name.c_str(), NS::UTF8StringEncoding);
    MTL::Function* fn = _library->newFunction(fn_str);
    if (!fn)
      throw std::runtime_error("Metal function not found: " + fn_name);

    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pso =
        _device.raw()->newComputePipelineState(fn, &error);
    fn->release();

    if (!pso) {
      std::string msg = "Failed to create pipeline for: " + fn_name;
      if (error)
        msg += std::string(": ") + error->localizedDescription()->utf8String();
      throw std::runtime_error(msg);
    }

    _pipeline_cache[fn_name] = pso;
    return pso;
  }

  using encode_fn_t =
      std::function<void(MTL::ComputeCommandEncoder*)>;

  void dispatch(const std::string& fn_name,
                MTL::Size grid_size,
                MTL::Size threadgroup_size,
                encode_fn_t encode) {
    auto pso = get_pipeline(fn_name);
    auto cmd_buf = _queue->commandBuffer();
    auto encoder = cmd_buf->computeCommandEncoder();

    encoder->setComputePipelineState(pso);
    encode(encoder);
    encoder->dispatchThreads(grid_size, threadgroup_size);
    encoder->endEncoding();

    cmd_buf->commit();
    cmd_buf->waitUntilCompleted();
  }

  void dispatch_1d(const std::string& fn_name,
                   std::size_t total_threads,
                   encode_fn_t encode) {
    auto pso = get_pipeline(fn_name);
    NS::UInteger max_tg = pso->maxTotalThreadsPerThreadgroup();
    NS::UInteger tg_size = std::min(max_tg, (NS::UInteger)256);

    // Round down to power of 2 for correct threadgroup-level algorithms
    // (scan, reduce use stride-halving which requires power-of-2 sizes).
    NS::UInteger po2 = 1;
    while (po2 * 2 <= tg_size)
      po2 *= 2;
    tg_size = po2;

    NS::UInteger n_groups = (total_threads + tg_size - 1) / tg_size;

    auto cmd_buf = _queue->commandBuffer();
    auto encoder = cmd_buf->computeCommandEncoder();
    encoder->setComputePipelineState(pso);
    encode(encoder);
    encoder->dispatchThreadgroups(MTL::Size(n_groups, 1, 1),
                                  MTL::Size(tg_size, 1, 1));
    encoder->endEncoding();
    cmd_buf->commit();
    cmd_buf->waitUntilCompleted();
  }

  MTL::CommandBuffer* begin_command() {
    return _queue->commandBuffer();
  }

  void synchronize() {
    auto cmd_buf = _queue->commandBuffer();
    cmd_buf->commit();
    cmd_buf->waitUntilCompleted();
  }

  void print_properties() const { _device.print_properties(); }
};

class multi_context_t {
  std::vector<context_t*> _contexts;
  std::vector<device_id_t> _devices;

 public:
  explicit multi_context_t(const std::string& shader_source,
                           device_id_t device = 0)
      : _devices(1, device) {
    _contexts.push_back(new context_t(shader_source, device));
  }

  multi_context_t(const std::string& shader_source,
                  std::vector<device_id_t> devices)
      : _devices(std::move(devices)) {
    for (auto& dev : _devices)
      _contexts.push_back(new context_t(shader_source, dev));
  }

  ~multi_context_t() {
    for (auto ctx : _contexts)
      delete ctx;
  }

  multi_context_t(const multi_context_t&) = delete;
  multi_context_t& operator=(const multi_context_t&) = delete;

  context_t* get_context(device_id_t device = 0) {
    return _contexts[device];
  }

  std::size_t size() const { return _contexts.size(); }
};

}  // namespace metal
}  // namespace gunrock
