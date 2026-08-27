# landscape-descriptors

JAX-based tools that construct topological descriptors of a high-dimensional loss landscape.

Mainly intended for use with the JAX-based differentiable simulator from the companion repository  `articulated-swimmers` (soon to be released).

- `critical_points` contains tools to search the landscape's critical points.
- `connectivity` contains tools to map the critical points connectivity and construct graph-based descriptors. 
- `optimizer` contains the gradient-based optimizer implementations utilized by the main tools.
