#pragma once

#include <gunrock/metal/context.hxx>
#include <gunrock/metal/buffer.hxx>

#include <functional>

namespace gunrock {
namespace metal {
namespace operators {
namespace parallel_for {

// Dispatch a named kernel over `count` threads.
// The encode_fn should bind all kernel buffers.
inline void execute(context_t& ctx,
                    const std::string& kernel_name,
                    std::size_t count,
                    context_t::encode_fn_t encode_fn) {
  if (count == 0)
    return;
  ctx.dispatch_1d(kernel_name, count, encode_fn);
}

}  // namespace parallel_for
}  // namespace operators
}  // namespace metal
}  // namespace gunrock
