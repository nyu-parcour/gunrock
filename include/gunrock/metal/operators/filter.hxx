#pragma once

#include <gunrock/metal/context.hxx>
#include <gunrock/metal/buffer.hxx>
#include <gunrock/metal/primitives.hxx>

namespace gunrock {
namespace metal {
namespace operators {
namespace filter {

// Stream compaction filter: removes invalid entries (negative values) from
// the frontier. Returns the number of valid elements.
inline std::size_t execute(context_t& ctx,
                           buffer_t<int>& input,
                           buffer_t<int>& output,
                           std::size_t count) {
  return primitives::compact(ctx, input, output, count);
}

}  // namespace filter
}  // namespace operators
}  // namespace metal
}  // namespace gunrock
