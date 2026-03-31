# Betweenness Centrality (BC) — Algorithm Outline & Bugfix

## Algorithm Overview

Gunrock's BC uses Brandes' algorithm, which has two phases:

1. **Forward BFS** — discover all vertices level-by-level from a source, recording shortest-path counts (`sigmas`) and BFS depth labels (`labels`).
2. **Backward sweep** — walk the BFS levels in reverse, accumulating dependency scores (`deltas`) and final BC values (`bc_values`).

Both phases use Gunrock's bulk-synchronous **advance** operator to traverse edges in parallel.

---

## Execution Flow

```
prepare_frontier()
  |
  v
loop() is called repeatedly until is_converged() returns true
  |
  |-- Phase 1: Forward BFS (forward == true)
  |     |
  |     |   while (frontier not empty):
  |     |     1. advance:  expand frontier along outgoing edges
  |     |                  - for each edge (src -> dst):
  |     |                    - atomically set labels[dst] = labels[src] + 1
  |     |                    - accumulate sigmas[dst] += sigmas[src]
  |     |                    - return true  (keep dst) if dst was unvisited
  |     |                    - return false (reject dst) otherwise
  |     |
  |     |     2. filter:   remove rejected vertices from output frontier  <-- THE FIX
  |     |
  |     |     3. converge: if output frontier is empty, switch to Phase 2
  |     |
  |     v
  |-- Phase 2: Backward sweep (forward == false)
  |     |
  |     |   while (depth > 0):
  |     |     1. advance:  walk each frontier level in reverse
  |     |                  - for each edge (src -> dst) where labels[src]+1 == labels[dst]:
  |     |                    - delta = sigmas[src] / sigmas[dst] * (1 + deltas[dst])
  |     |                    - accumulate deltas[src] += delta
  |     |                    - accumulate bc_values[src] += 0.5 * delta
  |     |                  - output type is `none` (no output frontier needed)
  |     |
  |     |     2. converge: if depth reaches 0, done
  |     v
  |
  v
is_converged(): returns true when both phases complete
```

---

## The Bug

### Gunrock's advance/filter contract

In Gunrock, the **advance** operator produces an output frontier that includes *every* edge destination — entries where the user's lambda returned `false` are written as `-1` (invalid sentinel). The expectation is that a **filter** step follows to compact the frontier, removing these `-1` entries.

Every other algorithm in Gunrock that uses advance follows this pattern:

```
advance  -->  filter  -->  check convergence
```

For example, BFS (`bfs.hxx`):
```cpp
operators::advance::execute_runtime(G, E, search, advance_load_balance, context);
operators::filter::execute_runtime(G, E, remove_invalids, filter_algorithm, context);
```

### What BC was doing wrong

BC's forward pass called advance but **never called filter**:

```cpp
// BEFORE (buggy)
while (true) {
    operators::advance::execute<...>(G, forward_op, in_frontier, out_frontier, ...);
    // <-- no filter here!
    this->depth++;
    if (is_forward_converged(context))
        break;
}
```

### Why this caused a segfault

1. Each advance iteration produced an output frontier full of `-1` entries mixed with valid vertices.
2. `is_forward_converged()` checks `out_frontier->is_empty()`, which tests `num_elements == 0`. Since `num_elements` counts *all* entries (valid + invalid), the frontier was never considered empty even when no new vertices were discovered.
3. On subsequent iterations, the `-1` entries were fed back into advance as input vertices. The advance kernel skips invalid inputs (`continue` on `-1`), but still allocates output space proportional to the frontier size.
4. The frontier grew unboundedly across iterations — a road network graph like `roadNet-CA` (1.97M vertices, diameter ~500+) meant hundreds of iterations of accumulating garbage, eventually causing an out-of-bounds memory access and segfault.

---

## The Fix

Added a `predicated` filter call after advance in the forward BFS loop (`bc.hxx`, lines 155-157):

```cpp
// AFTER (fixed)
auto remove_invalids =
    [] __host__ __device__(vertex_t const& vertex) -> bool {
  return vertex >= 0;  // keep all valid vertices; filter auto-removes -1 entries
};

while (true) {
    operators::advance::execute<...>(G, forward_op, in_frontier, out_frontier, ...);

    // Remove invalid (-1) entries from the output frontier.
    operators::filter::execute<operators::filter_algorithm_t::predicated>(
        G, remove_invalids, out_frontier, out_frontier, context);

    this->depth++;
    if (is_forward_converged(context))
        break;
}
```

The `predicated` filter uses `thrust::copy_if` internally. It automatically discards any element that is the invalid sentinel (`-1`), and only passes valid vertices to the user lambda. Since we want to keep all valid vertices, the lambda simply returns `true`.

### Why the backward pass doesn't need this fix

The backward sweep uses `advance_io_type_t::none` for its output, meaning no output frontier is produced — it just accumulates into `deltas` and `bc_values` as a side effect. Convergence is controlled by decrementing `depth` to 0, not by checking frontier emptiness.

---

## Bug 2: Frontier Buffer Overflow (segfault)

### What was happening

The BC enactor allocates a fixed number of frontier buffers at construction time:

```cpp
props.number_of_frontier_buffers = 1000;  // XXX: hack!
```

Each BFS level uses a separate frontier (`this->frontiers[depth]` and `this->frontiers[depth + 1]`). On road networks, the BFS diameter from certain source vertices exceeds 999 levels. When `depth + 1 >= 1000`, the access to `this->frontiers[1000]` is an out-of-bounds read on the host-side `thrust::host_vector`, causing a segfault.

### Why it was intermittent

The source vertex is chosen **randomly** when `--src` is not specified. Most sources produce BFS trees with depth < 1000, but some sources on large-diameter graphs (e.g. `roadNet-CA`) produce deeper trees that overflow the buffer.

### The fix

Dynamically grow the frontier buffer when the BFS depth approaches the current capacity:

```cpp
while (true) {
    // Grow frontier buffer if BFS depth exceeds current capacity.
    if (this->depth + 2 > this->frontiers.size())
      this->frontiers.resize(this->depth + 2);

    auto in_frontier = &(this->frontiers[this->depth]);
    auto out_frontier = &(this->frontiers[this->depth + 1]);
    // ...
}
```

---

## Bug 3: NaN in BC values (float overflow in sigmas)

### What was happening

The `sigmas` array stores shortest-path counts as `weight_t` (`float`). On graphs where many shortest paths converge — especially with deep BFS trees — sigma values overflow `float`'s range (~3.4e38) and become `Inf`.

In the backward pass, when both `sigmas[src]` and `sigmas[dst]` are `Inf`:

```cpp
auto update = sigmas[src] / sigmas[dst] * (1 + deltas[dst]);
//             Inf     /    Inf       =  NaN
```

`Inf / Inf = NaN`, which then propagates through all `deltas` and `bc_values` via the atomic adds.

### The fix

Guard against non-finite update values in the backward operator:

```cpp
auto update = sigmas[src] / sigmas[dst] * (1 + deltas[dst]);
if (!isfinite(update))
  return false;
```

When float precision is exhausted, the contribution is skipped. This is lossy but prevents NaN corruption of the entire output.

A more precise fix would be to use `double` for `sigmas` and `deltas`, but these types are currently tied to the graph's `weight_t`.

---

## Verified

- `chesapeake.mtx` (39 vertices): completes correctly
- `roadNet-CA.mtx` (1.97M vertices, 5.5M edges): completes in ~31ms (previously segfaulted)
- `roadNet-CA.mtx` with source 186714 (deep BFS, ~1000+ levels): completes without NaN (previously produced NaN)
- 20 consecutive runs with random sources on `roadNet-CA`: 0 segfaults, 0 NaN
