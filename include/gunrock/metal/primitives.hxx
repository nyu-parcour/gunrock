#pragma once

#include <gunrock/metal/context.hxx>
#include <gunrock/metal/buffer.hxx>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace gunrock {
namespace metal {
namespace primitives {

inline void fill(context_t& ctx,
                 buffer_t<int>& buf,
                 int value,
                 std::size_t count) {
  buffer_t<uint32_t> count_buf(ctx.device().raw(), 1);
  buffer_t<int> val_buf(ctx.device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(count);
  val_buf[0] = value;

  ctx.dispatch_1d("fill_int", count, [&](MTL::ComputeCommandEncoder* enc) {
    enc->setBuffer(buf.raw(), 0, 0);
    enc->setBuffer(count_buf.raw(), 0, 1);
    enc->setBuffer(val_buf.raw(), 0, 2);
  });
}

inline void fill(context_t& ctx,
                 buffer_t<uint32_t>& buf,
                 uint32_t value,
                 std::size_t count) {
  buffer_t<uint32_t> count_buf(ctx.device().raw(), 1);
  buffer_t<uint32_t> val_buf(ctx.device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(count);
  val_buf[0] = value;

  ctx.dispatch_1d("fill_uint", count, [&](MTL::ComputeCommandEncoder* enc) {
    enc->setBuffer(buf.raw(), 0, 0);
    enc->setBuffer(count_buf.raw(), 0, 1);
    enc->setBuffer(val_buf.raw(), 0, 2);
  });
}

inline void fill(context_t& ctx,
                 buffer_t<float>& buf,
                 float value,
                 std::size_t count) {
  buffer_t<uint32_t> count_buf(ctx.device().raw(), 1);
  buffer_t<float> val_buf(ctx.device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(count);
  val_buf[0] = value;

  ctx.dispatch_1d("fill_float", count, [&](MTL::ComputeCommandEncoder* enc) {
    enc->setBuffer(buf.raw(), 0, 0);
    enc->setBuffer(count_buf.raw(), 0, 1);
    enc->setBuffer(val_buf.raw(), 0, 2);
  });
}

// Multi-pass Blelloch exclusive prefix scan.
// Supports up to 256*256 = 65536 elements per call.
inline void exclusive_scan(context_t& ctx,
                           buffer_t<uint32_t>& input,
                           buffer_t<uint32_t>& output,
                           std::size_t count) {
  if (count == 0)
    return;

  const uint32_t block_size = 256;
  uint32_t n_blocks = (static_cast<uint32_t>(count) + block_size - 1) / block_size;

  buffer_t<uint32_t> partial_sums(ctx.device().raw(), n_blocks);
  buffer_t<uint32_t> count_buf(ctx.device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(count);

  // Pass 1: per-block scan
  {
    auto pso = ctx.get_pipeline("scan_per_block");
    auto cmd_buf = ctx.queue()->commandBuffer();
    auto enc = cmd_buf->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(input.raw(), 0, 0);
    enc->setBuffer(output.raw(), 0, 1);
    enc->setBuffer(partial_sums.raw(), 0, 2);
    enc->setBuffer(count_buf.raw(), 0, 3);
    enc->dispatchThreads(MTL::Size(n_blocks * block_size, 1, 1),
                         MTL::Size(block_size, 1, 1));
    enc->endEncoding();
    cmd_buf->commit();
    cmd_buf->waitUntilCompleted();
  }

  // Pass 2: scan the partial sums (single threadgroup, handles up to 256 blocks)
  if (n_blocks > 1) {
    buffer_t<uint32_t> nblocks_buf(ctx.device().raw(), 1);
    nblocks_buf[0] = n_blocks;

    auto pso = ctx.get_pipeline("scan_partial_sums");
    auto cmd_buf = ctx.queue()->commandBuffer();
    auto enc = cmd_buf->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(partial_sums.raw(), 0, 0);
    enc->setBuffer(nblocks_buf.raw(), 0, 1);
    enc->dispatchThreads(MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
    enc->endEncoding();
    cmd_buf->commit();
    cmd_buf->waitUntilCompleted();

    // Pass 3: propagate to global output
    auto pso2 = ctx.get_pipeline("scan_propagate");
    auto cmd_buf2 = ctx.queue()->commandBuffer();
    auto enc2 = cmd_buf2->computeCommandEncoder();
    enc2->setComputePipelineState(pso2);
    enc2->setBuffer(output.raw(), 0, 0);
    enc2->setBuffer(partial_sums.raw(), 0, 1);
    enc2->setBuffer(count_buf.raw(), 0, 2);
    enc2->dispatchThreads(MTL::Size(n_blocks * block_size, 1, 1),
                          MTL::Size(block_size, 1, 1));
    enc2->endEncoding();
    cmd_buf2->commit();
    cmd_buf2->waitUntilCompleted();
  }
}

inline uint32_t reduce_sum(context_t& ctx,
                           buffer_t<uint32_t>& input,
                           std::size_t count) {
  buffer_t<uint32_t> result(ctx.device().raw(), 1);
  result[0] = 0;
  buffer_t<uint32_t> count_buf(ctx.device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(count);

  ctx.dispatch_1d(
      "reduce_sum_uint", count, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(input.raw(), 0, 0);
        enc->setBuffer(result.raw(), 0, 1);
        enc->setBuffer(count_buf.raw(), 0, 2);
      });

  return result[0];
}

inline int32_t reduce_sum(context_t& ctx,
                          buffer_t<int32_t>& input,
                          std::size_t count) {
  buffer_t<int32_t> result(ctx.device().raw(), 1);
  result[0] = 0;
  buffer_t<uint32_t> count_buf(ctx.device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(count);

  ctx.dispatch_1d(
      "reduce_sum_int", count, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(input.raw(), 0, 0);
        enc->setBuffer(result.raw(), 0, 1);
        enc->setBuffer(count_buf.raw(), 0, 2);
      });

  return result[0];
}

// Stream compaction: keep elements >= 0 (removes -1 sentinel invalid entries).
// Returns the number of valid elements written to output.
inline std::size_t compact(context_t& ctx,
                           buffer_t<int>& input,
                           buffer_t<int>& output,
                           std::size_t count) {
  if (count == 0)
    return 0;

  buffer_t<uint32_t> marks(ctx.device().raw(), count);
  buffer_t<uint32_t> offsets(ctx.device().raw(), count);
  buffer_t<uint32_t> count_buf(ctx.device().raw(), 1);
  count_buf[0] = static_cast<uint32_t>(count);

  // Step 1: mark valid elements
  ctx.dispatch_1d("mark_valid", count, [&](MTL::ComputeCommandEncoder* enc) {
    enc->setBuffer(input.raw(), 0, 0);
    enc->setBuffer(marks.raw(), 0, 1);
    enc->setBuffer(count_buf.raw(), 0, 2);
  });

  // Step 2: exclusive scan of marks -> offsets
  exclusive_scan(ctx, marks, offsets, count);

  // Step 3: compute total valid count
  uint32_t last_mark = marks[count - 1];
  uint32_t last_offset = offsets[count - 1];
  uint32_t total_valid = last_offset + last_mark;

  if (total_valid == 0)
    return 0;

  // Step 4: scatter valid elements
  ctx.dispatch_1d(
      "scatter_valid", count, [&](MTL::ComputeCommandEncoder* enc) {
        enc->setBuffer(input.raw(), 0, 0);
        enc->setBuffer(output.raw(), 0, 1);
        enc->setBuffer(marks.raw(), 0, 2);
        enc->setBuffer(offsets.raw(), 0, 3);
        enc->setBuffer(count_buf.raw(), 0, 4);
      });

  return total_valid;
}

}  // namespace primitives
}  // namespace metal
}  // namespace gunrock
