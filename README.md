# landscape-descriptors

JAX-based tools that implement a topological descriptor of high-dimensional loss landscapes.

Mainly intended for use with the JAX-based differentiable simulator from the companion repository  `articulated-swimmers` (soon to be released).

- `critical_points` contains tools to search the landscape's critical points.
- `connectivity` contains tools to map the critical points connectivity and construct graph-based descriptors. 
- `optimizer` contains the gradient-based optimizer implementations utilized by the main tools.
