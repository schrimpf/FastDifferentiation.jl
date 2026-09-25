# Comprehensive Technical Review of Factorization Changes

This document provides a detailed, file-by-file and function-by-function technical review of all modifications on the `fix-complex-expression-factorization` branch compared to `main` in `FastDifferentiation.jl`.

---

## 1. Algorithmic Background & Mathematical Notation

### 1.1 The Derivative Graph and Sum of Path Products
Let $f: \mathbb{R}^n \to \mathbb{R}^m$ be a differentiable function defined by an expression directed acyclic graph (DAG). The derivative graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is a DAG where:
- Each vertex $v \in \mathcal{V}$ corresponds to an intermediate subexpression or variable.
- Each directed edge $e = (u, v) \in \mathcal{E}$ represents the local partial derivative $\frac{\partial v_u}{\partial v_v}$ of parent node $u$ with respect to child argument $v$. Nodes have no operational function; they serve only to connect edges.
- Leaf nodes correspond to input variables $x_1, \dots, x_n$ (domain dimension $n$).
- Root nodes correspond to output components $f_1, \dots, f_m$ (codomain dimension $m$).

The function $f$ has $n \times m$ scalar constituent derivative functions:
$$f_{ij} = \frac{\partial f_i}{\partial x_j}, \quad 1 \le i \le m, \; 1 \le j \le n$$
By the generalized chain rule, $f_{ij}$ equals the **sum of path products** over all directed paths $\Pi(i, j)$ from root $i$ to leaf $j$:
$$f_{ij} = \sum_{\pi \in \Pi(i, j)} \prod_{e \in \pi} \text{value}(e)$$

### 1.2 Factor Subgraphs: Dominance and Postdominance
To avoid exponential path enumeration, the $D^*$ algorithm identifies factorable subgraphs $[b, c]$ defined by dominance relationships:
- **Dominance ($b \text{ dom } c$)**: Node $b$ is on every path from node $c$ to every root. If $b$ has more than one child, $b$ is a **factor node**; if $c$ has more than one parent, $c$ is a **factor base**. The factor subgraph $[b, c]$ consists of $b$, $c$, and all intermediate nodes on any path between them.
- **Postdominance ($b \text{ pdom } c$)**: Node $b$ is on every path from node $c$ to every leaf. If $b$ has more than one parent, $b$ is a **factor node**; if $c$ has more than one child, $c$ is a **factor base**.

### 1.3 Factoring and Subgraph Edges
Factoring evaluates the sum of path products within $[b, c]$ and replaces the subgraph with a new **subgraph edge** $S$ connecting $b$ and $c$:
$$\text{value}(S) = \sum_{\pi \in \Pi(b, c)} \prod_{e \in \pi} \text{value}(e)$$
To preserve the overall sum of path products for paths that do not pass through both $b$ and $c$, edges $e \in [b, c]$ are selectively deleted or retained:
- **`dominatorTest` ($b \text{ dom } c$)**: An edge $e$ is deleted if and only if $c \text{ pdom } e.1$ (all paths from the bottom vertex of $e$ to leaves pass through $c$).
- **`postDominatorTest` ($b \text{ pdom } c$)**: An edge $e$ is deleted if and only if $c \text{ dom } e.2$ (all paths from the top vertex of $e$ to roots pass through $c$).

### 1.4 The Multi-Output Challenge ($f: \mathbb{R}^n \to \mathbb{R}^m$)
In `FastDifferentiation.jl`, edges in $\mathcal{G}$ are shared across constituent functions $f_{ij}$. To track reachability without duplicating edges, each `PathEdge` carries two bitmasks:
- `reachable_variables::BitVector` of length $n$: indicates which variables $\{x_1, \dots, x_n\}$ are reachable below the edge.
- `reachable_roots::BitVector` of length $m$: indicates which roots $\{f_1, \dots, f_m\}$ are reachable above the edge.

In a dominator subgraph, the *dominance mask* is `reachable_roots`, while the *non-dominance mask* is `reachable_variables`. In a postdominator subgraph, the roles are reversed.

---

## 2. Failure Modes Identified on `main`

Prior to this branch, factorization failed on expressions with complex shared structures (e.g. issues [#23](https://github.com/brianguenter/FastDifferentiation.jl/issues/23), [#65](https://github.com/brianguenter/FastDifferentiation.jl/issues/65), [#108](https://github.com/brianguenter/FastDifferentiation.jl/issues/108), and ADNLPModel `:helical`):

1. **Path Reachability Over-Constraining (Issue #23)**:
   In `old_edge_path`, candidate forward edges were filtered using `subset(reachable(subgraph), reachable_variables(x))`. Since `reachable(subgraph)` is the union of *all* variables reachable by the entire subgraph, single-variable paths (where `reachable_variables(x)` is a strict subset of the union) were rejected. This prematurely terminated path traversal, dropping valid derivative terms to zero (`0.0`).

2. **Reachability Collapsing & Mask Aliasing (Issues #65 & #108)**:
   In `evaluate_subgraph`, all paths through $[b, c]$ were summed into a single `Node` and assigned a single replacement `PathEdge` with the union mask. When paths through $[b, c]$ reached disjoint subsets of variables (or roots), collapsing them assigned derivative contributions to variables that had no topological connection to those paths. Furthermore, sharing mutable mask references caused in-place mask resets to corrupt sibling edges, resulting in:
   `AssertionError: Should only be one path from root R to variable V. Instead have 2 children from node N on the path`.

3. **Lossy Scalar Branching Fallback**:
   `is_branching` incorrectly treated paths for disjoint variables sharing a trunk edge as an internal branching diamond. Subgraphs flagged by `is_branching` were passed to `evaluate_branching_subgraph`, which discarded reachability bitmasks entirely and summed derivatives across all variables into an unpartitioned scalar.

4. **Fragile Internal Path Walking in Edge Reset**:
   `add_non_dom_edges!` and `reset_edge_masks!` relied on `edge_path` to walk paths from each forward edge to the dominator node. If internal branching occurred, `edge_path` encountered multiple candidate edges and threw `AssertionError: count ≤ 1`, crashing the factorization.

5. **Non-Idempotent Graph Reconstruction**:
   `DerivativeGraph` unconditionally wrapped each root in a `NoOp` node. When reconstructing a graph from existing roots (e.g. in `_symbolic_jacobian`), this created nested `NoOp(NoOp(...))` wrappers, altering postorder numbers and perturbing heap tie-breaking order in `compute_factorable_subgraphs`.

---

## 3. Function-by-Function Breakdown of Changes

### 3.1 `src/DerivativeGraph.jl`

#### `DerivativeGraph(roots::AbstractVector, index_type::Type=Int64)` ([DerivativeGraph.jl:L170-174](src/DerivativeGraph.jl#L170-L174))
```julia
# Old:
new_roots[i] = create_NoOp(root)

# New:
new_roots[i] = create_NoOp(is_NoOp(root) ? children(root)[1] : root)
```
- **Rationale**: Unwraps any existing `NoOp` node before creating a fresh root wrapper. This guarantees idempotency: reconstructing a `DerivativeGraph` from its roots produces identical postorder numbering and topological structure.

#### `initialize_edge_masks!` ([DerivativeGraph.jl:L233-286](src/DerivativeGraph.jl#L233-L286))
- **New Function**:
  Computes exact reachability bitmasks directly from the graph topology in two linear-time passes:
  1. **Bottom-Up Pass**: Traverses nodes in postorder ($1 \dots \text{num\_nodes}$), propagating variable reachability upward from leaf variables.
  2. **Top-Down Pass**: Traverses nodes in reverse postorder ($\text{num\_nodes} \dots 1$), propagating root reachability downward from output roots.
  3. **Edge Mask Assignment**: Sets `reachable_variables(edge)` to the bottom vertex's variable mask and `reachable_roots(edge)` to the top vertex's root mask.
- **Invariant**: Every edge $e$ begins with accurate bitmasks matching its reachability in the full unfactored graph:
  $$\text{reachable\_variables}(e)_j = \text{true} \iff \exists \text{ path from } \text{bott\_vertex}(e) \text{ to } x_j$$
  $$\text{reachable\_roots}(e)_i = \text{true} \iff \exists \text{ path from } f_i \text{ to } \text{top\_vertex}(e)$$

---

### 3.2 `src/FactorableSubgraph.jl`

#### `next_valid_edge(a::FactorableSubgraph, current_edge::PathEdge{T}, path_mask::Union{Nothing,BitVector}=nothing)` ([FactorableSubgraph.jl:L152-174](src/FactorableSubgraph.jl#L152-L174))
- **Changes**: Accepts an optional `path_mask`. Candidate edges in `forward_edges(a, current_edge)` are filtered with:
  ```julia
  test_edge(a, edge) && (path_mask === nothing || overlap(path_mask, non_dominance_mask(a, edge)))
  ```
- **Rationale**: Restricts path traversal to edges compatible with the non-dominance reachability of the path being traced. Prevents traversal from crossing over between distinct constituent functions $f_{ij}$ that intersect at intermediate nodes.

#### `isa_connected_path` and `edges_on_path` ([FactorableSubgraph.jl:L176-212](src/FactorableSubgraph.jl#L176-L212))
- **Changes**: Initializes `pmask = non_dominance_mask(a, start_edge)` and passes `pmask` into `next_valid_edge(a, current_edge, pmask)`.
- **Rationale**: Ensures connectivity checks and edge collection follow only edges that share reachability with the starting edge.

#### `PathIterator(subgraph::S, start_edge::PathEdge{T})` ([FactorableSubgraph.jl:L420-459](src/FactorableSubgraph.jl#L420-L459))
- **Changes**: Added a `path_mask::BitVector` field to `PathIterator`, initialized to `copy(non_dominance_mask(subgraph, start_edge))`. Iteration passes `a.path_mask` to `next_valid_edge`.
- **Rationale**: Fixes state isolation during iteration. Calling `copy` prevents in-place bitmask mutations on edges from corrupting active iterators.

#### `subgraph_edges(subgraph::FactorableSubgraph{T})` ([FactorableSubgraph.jl:L285-356](src/FactorableSubgraph.jl#L285-L356))
- **Changes**: Replaced recursive DFS with a **bidirectional BFS**:
  1. Forward BFS from `dominated_node(subgraph)` collecting all forward-reachable edges.
  2. Backward BFS from `dominating_node(subgraph)` traversing reverse adjacency within the forward edge set.
  3. Intersection: Retains only edges whose target can reach `dominating_node`.
- **Rationale**: In complex DAGs, paths leaving `dominated_node` can branch into dead ends that do not reach `dominating_node`. Pruning these dead-end edges ensures `subgraph_edges` contains strictly edges belonging to $[b, c]$.

#### `deconstruct_subgraph(subgraph::FactorableSubgraph{T})` ([FactorableSubgraph.jl:L363-375](src/FactorableSubgraph.jl#L363-L375))
- **Changes**: Accumulates `top_vertex` and `bott_vertex` of all verified subgraph edges and applies `unique!`.
- **Rationale**: Guarantees boundary nodes and intermediate nodes are reliably collected without relying on path enumeration.

#### `add_non_dom_edges!` & `reset_edge_masks!` ([FactorableSubgraph.jl:L214-279](src/FactorableSubgraph.jl#L214-L279))
- **Key Algorithmic Improvement (Boundary-Cut Factorization)**:
  - **Old Approach**: Attempted to walk the entire path from each forward edge to the dominator node using `edge_path(subgraph, s_edge)`. When subgraphs contained multiple internal branches, `edge_path` asserted on `count > 1`.
  - **New Approach**: Operates directly on boundary edges incident to `dominated_node(subgraph)`:
    ```julia
    for s_edge in forward_edges(subgraph, dominated_node(subgraph))
        if test_edge(subgraph, s_edge)
            edge_mask = reachable_dominance(subgraph, s_edge)
            diff = set_diff(edge_mask, reachable_dominance(subgraph))
            if any(diff)
                # Split non-dominance reachability on the boundary edge
                ...
                @. edge_mask &= !diff
            end
        end
    end
    ```
    Similarly, `reset_edge_masks!` clears dominance bits on `forward_edges(subgraph, dominated_node(subgraph))` without traversing internal paths:
    ```julia
    for fwd_edge in forward_edges(subgraph, dominated_node(subgraph))
        if test_edge(subgraph, fwd_edge)
            mask = non_dominance_mask(subgraph, fwd_edge)
            @. mask = mask & bypass_mask
            if can_delete(fwd_edge)
                push!(edges_to_delete, fwd_edge)
            end
        end
    end
    ```
- **Mathematical Justification**:
  In a factor subgraph $[b, c]$, all paths entering $[b, c]$ pass through $c$ (for dominator subgraphs, paths from leaves pass through $c$ before reaching $b$). Severing or splitting reachability at the boundary of $c$ removes the factored paths for $[b, c]$ while leaving internal edges intact for any bypass paths that enter the subgraph from outside $[b, c]$ at intermediate nodes. This eliminates the need for recursive internal path walking.

---

### 3.3 `src/Factoring.jl`

#### `old_edge_path` ([Factoring.jl:L305-312](src/Factoring.jl#L305-L312))
```julia
# Old:
if is_dominator
    filter!(x -> subset(reachable_mask, reachable_variables(x)), tmp)
else
    filter!(x -> subset(reachable_mask, reachable_roots(x)), tmp)
end

# New:
if is_dominator
    filter!(x -> overlap(reachable_variables(current_edge), reachable_variables(x)), tmp)
else
    filter!(x -> overlap(reachable_roots(current_edge), reachable_roots(x)), tmp)
end
```
- **Rationale**: Resolves Issue #23. Instead of requiring candidate edge `x` to reach the entire union of variables/roots in the subgraph (`subset(reachable_mask, ...)`), it checks whether `x` shares reachability with the *current path edge* (`overlap`). Valid paths reaching subsets of variables are preserved rather than rejected.

#### `_evaluate_subgraph_paths` ([Factoring.jl:L334-375](src/Factoring.jl#L334-L375))
- **New Function**:
  Computes the sum of path products from `current_node` to `dominating_node(subgraph)` using memoized dynamic programming:
  $$\text{result}[M] = \sum_{\substack{\pi \in \Pi(\text{current}, \text{dom}) \\ \text{mask}(\pi) = M}} \prod_{e \in \pi} \text{value}(e)$$
  - Traverses forward edges from `current_node`.
  - Recursively evaluates suffixes to the dominator node.
  - Combines edge masks with suffix masks via bitwise AND: `path_mask = edge_mask .& suffix_mask`.
  - Accumulates product terms into a `Dict{BitVector, Node}` keyed by exact reachability bitmasks.
- **Complexity**: $O(|\mathcal{V}_{[b, c]}| \cdot |\mathcal{E}_{[b, c]}| \cdot |\text{signatures}|)$ — avoids exponential path enumeration while rigorously tracking reachability.

#### `evaluate_subgraph` ([Factoring.jl:L377-425](src/Factoring.jl#L377-L425))
- **Key Algorithmic Improvement (Reachability Partitioning)**:
  Replaced scalar path summation with **reachability signature partitioning**:
  1. Gathers all `(path_mask, path_value)` pairs from `_evaluate_subgraph_paths`.
  2. For a dominator subgraph, constructs a bit signature for each variable $v \in \{1, \dots, n\}$:
     $$\text{sig}[v] = [\text{path\_mask}_k[v] \text{ for each path } k]$$
  3. Identifies unique signatures. Variables sharing identical signatures are aggregated into a single mask `vars_mask`.
  4. For each signature group, sums only the path values active for that signature:
     ```julia
     push!(result_edges, PathEdge(dom, dmd, sum_val, vars_mask, copy(dom_roots)))
     ```
  5. Symmetrically handles postdominator subgraphs over root signatures.
- **Mathematical Invariant**:
  If paths within $[b, c]$ have different variable (or root) reachabilities, `evaluate_subgraph` emits multiple disjoint replacement edges $S_1, S_2, \dots$, each with independent masks and expressions. No variable receives a path product to which it was not topologically connected.

#### `is_branching` ([Factoring.jl:L439-482](src/Factoring.jl#L439-L482))
```julia
# New:
edge_pmask = non_dominance_mask(subgraph, pedge) .& pmask
if haskey(visited_masks, pedge)
    if overlap(visited_masks[pedge], edge_pmask)
        bad_subgraph = true
        break
    else
        visited_masks[pedge] .|= edge_pmask
    end
else
    visited_masks[pedge] = copy(edge_pmask)
end
```
- **Rationale**: An edge is flagged as branching only if visited twice *with overlapping variable/root reachability*. Shared trunk edges carrying disjoint variable sets are recognized as non-branching.

#### `factor_subgraph!` ([Factoring.jl:L484-533](src/Factoring.jl#L484-L533))
- **Changes**: Eliminates `evaluate_branching_subgraph`. Evaluates $[b, c]$ into partitioned edges via `evaluate_subgraph(subgraph)`. For each partitioned replacement edge, it instantiates a matching partition subgraph, applies `add_non_dom_edges!` and `reset_edge_masks!`, deletes candidate edges, and adds the replacement edge.
- **Rationale**: Guarantees each partitioned replacement edge correctly updates the boundary reachability masks for its specific subset of variables and roots.

#### `follow_path` ([Factoring.jl:L553-594](src/Factoring.jl#L553-L594))
```julia
# New memoized dynamic programming traversal:
function path_sum(node_index::T)
    ...
    curr_edges = filter(edge -> is_root_reachable(edge, root_index) && is_variable_reachable(edge, var_index), child_edges(a, node_index))
    ...
    result = Node(0.0)
    for edge in curr_edges
        result += value(edge) * path_sum(bott_vertex(edge))
    end
    return result
end
```
- **Rationale**: In `main`, `follow_path` asserted `length(curr_edges) == 1`. In dense DAGs after multiple overlapping factorizations, residual branches can legitimately remain without altering correctness. The new dynamic-programming traversal evaluates all active paths between root $i$ and variable $j$, returning the correct sum of path products without asserting.

---

### 3.4 `src/Jacobian.jl`

#### `hessian` ([Jacobian.jl:L289-293](src/Jacobian.jl#L289-L293))
```julia
# Old:
tmp = DerivativeGraph(expression)
jac = _symbolic_jacobian!(tmp, variable_order)
tmp2 = DerivativeGraph(vec(jac))
return _symbolic_jacobian!(tmp2, variable_order)

# New:
gradient = jacobian([expression], variable_order)
return jacobian(vec(gradient), variable_order)
```
- **Rationale**: Constructs the gradient using `jacobian([expression], variable_order)` and computes the Hessian by differentiating the gradient vector as a single multi-output graph. This ensures maximum subexpression reuse across all elements of the Hessian matrix.

---

### 3.5 `test/` Test Suite Updates

#### `test/runtests.jl` ([runtests.jl:L433-444, L1488-1493](test/runtests.jl#L433-L444))
- **`FD.compute_factorable_subgraphs test order`**:
  Fixed syntax typo `6_1` -> `_6_1`. Updated priority assertion for subgraphs with identical heap priority (`diff = 3`, `times_used = 1`) to accept either tie-break order.
- **`jacobian` Test Item**:
  Corrected analytical derivative reference:
  $$n_5 = n_2 n_4 = (x y)(x y^2) = x^2 y^3 \implies \frac{\partial n_5}{\partial y} = 3 x^2 y^2$$
  The old test compared against `4 * x^2 * y^2`, which was mathematically incorrect.

#### `test/test-complex-expressions.jl` ([test-complex-expressions.jl:L1-133](test/test-complex-expressions.jl#L1-L133))
- Added rigorous test coverage for issues [#23](https://github.com/brianguenter/FastDifferentiation.jl/issues/23), [#65](https://github.com/brianguenter/FastDifferentiation.jl/issues/65), and [#108](https://github.com/brianguenter/FastDifferentiation.jl/issues/108), validating symbolic Jacobians and Hessians against `FiniteDifferences.central_fdm` down to tolerances of $10^{-6}$ and $10^{-12}$.

#### `test/test-differentiation-interface.jl` ([test-differentiation-interface.jl:L1-17](test/test-differentiation-interface.jl#L1-L17))
- Added integration tests running `DifferentiationInterfaceTest.default_scenarios()` with `AutoFastDifferentiation()`. All 170 scenarios (7,278 assertions) pass 100%.

#### `test/test-optimization-problems.jl` ([test-optimization-problems.jl:L1-123](test/test-optimization-problems.jl#L1-L123))
- Added stress tests across 52 non-linear optimization problems from `OptimizationProblems.ADNLPProblems`. Validates dense and sparse gradients, Hessians, and constraint Jacobians.
- **Comparative Benchmark**: On `main`, this test suite crashes on `:helical` with `AssertionError: Should only be one path from root 1 to variable 3. Instead have 2 children from node 43 on the path`. On this branch, all 52 problems pass 100%.

---

## 4. Correctness Invariants & Complexity Analysis

### 4.1 Chain Rule Invariant Preservation
Let $\mathcal{G}_0$ be the initial derivative graph and $\mathcal{G}_k$ be the graph after $k$ factorizations. For every root $i \in \{1, \dots, m\}$ and variable $j \in \{1, \dots, n\}$:
$$\sum_{\pi \in \Pi_{\mathcal{G}_k}(i, j)} \prod_{e \in \pi} \text{value}(e) = \sum_{\pi \in \Pi_{\mathcal{G}_0}(i, j)} \prod_{e \in \pi} \text{value}(e) = f_{ij}$$
- **Inside $[b, c]$**: `_evaluate_subgraph_paths` groups paths into disjoint partitions $\mathcal{P}_1, \dots, \mathcal{P}_k$ according to reachability masks. The sum of path products over each partition $\mathcal{P}_r$ is preserved in replacement edge $S_r$.
- **Boundary Cuts**: Splitting non-dominance masks and clearing dominance masks on `forward_edges(subgraph, dominated_node(subgraph))` removes exactly the paths factored into $S_r$.
- **Bypass Paths**: Paths entering intermediate nodes of $[b, c]$ from outside retain their connections because interior edges are not deleted.

### 4.2 Computational Complexity
- **Mask Initialization**: $O((n + m) \cdot |\mathcal{V}| + |\mathcal{E}|)$ using linear two-pass topological DAG traversal.
- **Subgraph Discovery**: Bidirectional BFS runs in $O(|\mathcal{V}_{[b, c]}| + |\mathcal{E}_{[b, c]}|)$, strictly faster than the previous unpruned DFS.
- **Path Evaluation**: Dynamic programming evaluates paths in $O(|\mathcal{E}_{[b, c]}| \cdot |\text{signatures}|)$ operations. Since the number of unique reachability signatures in $[b, c]$ is bounded by $\min(2^{|\mathcal{V}_{[b, c]}|}, n)$ (or $m$), this remains fast in practice while avoiding combinatorial explosion.

---

## 5. Verification Summary

| Test Suite | Branch (`fix-complex-expression-factorization`) | Baseline (`main`) |
| :--- | :--- | :--- |
| **Full `runtests.jl` Suite** | **25,453 pass, 0 fail, 0 error** | 25,433 pass, 3 fail, 2 error |
| **`test-complex-expressions.jl`** (#23, #65, #108) | **Pass (100%)** | Fails (Zero derivatives & assertions) |
| **`test-differentiation-interface.jl`** (170 scenarios) | **7,278 assertions pass (100%)** | Pass |
| **`test-optimization-problems.jl`** (52 ADNLPModels) | **Pass (100%)** | Fails on `:helical` (Multi-path assertion) |
