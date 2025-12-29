# --- Constants (Inferred from C++ context) ---
# Critical Point Types
SADDLE = 'saddle'
MINIMUM = 'minimum'
MAXIMUM =  'maximum'

class MergeTree:
    """
    A class implementing the Join Tree (Merge Tree for sublevel sets) algorithm.
    It utilizes a Disjoint Set Union (DSU) structure to manage component merging
    and build the structure of the tree.
    
    The critical point types are provided as input, simplifying the classification logic.
    """

    def __init__(self, sorted_vertices: list[int], neighbor_map: dict[int, list[int]], critical_points_map: dict[int, int]):
        """
        Initializes the MergeTree with actual input data.
        
        Args:
            sorted_vertices: A list of all vertex IDs, sorted by their function value
                             in ascending order (i.e., sv[0] is the global minimum).
            neighbor_map: A dictionary mapping each vertex ID to a list of its neighbors.
                          {vertex_id: [neighbor1_id, neighbor2_id, ...]}
            critical_points_map: A dictionary mapping *every* input vertex ID to its critical 
                                 point type (MAXIMUM, MINIMUM, or SADDLE).
        """
        
        self.sv = sorted_vertices
        self.neighbor_map = neighbor_map
        self.no_vertices = len(self.sv)
        
        # Critical Points (critical_pts): Directly loaded from the input map.
        # All vertices in self.sv are assumed to be present in this map.
        self.critical_pts = critical_points_map
        
        # --- Pre-computation for O(1) function value comparison ---
        # Map vertex ID to its index in the sorted list (function value rank)
        self._vertex_rank = {v: i for i, v in enumerate(self.sv)}

        # --- DSU and Tree Structure Initialization ---
        
        # DSU structure (nodes in C++): self.parent[i] stores the parent of vertex i.
        max_vertex_id = max(self.sv) if self.sv else -1
        # Initialize DSU: each vertex is its own component.
        self.parent = list(range(max_vertex_id + 1))
        
        # cpMap: Maps component representative (root of DSU) to the actual critical point vertex.
        # This tracks the active critical point for each component during processing.
        self.cp_map = {}
        
        # prev: Maps a component's critical point (to) to the vertex (from) that merged it. (Tree edges)
        self.prev = {}
            
        # Member variables that are set by the computation (initializing to default values)
        self.new_root = -1

    # --- Data Access Helpers ---
    
    def get_star(self, v: int) -> list[int]:
        """Returns the list of neighbors for vertex v (C++ data->getStar)."""
        return self.neighbor_map.get(v, [])

    def is_upper_link(self, v1: int, v2: int) -> bool:
        """
        Checks if f(v1) < f(v2) (C++ data->lessThan(v1, v2)).
        A higher rank (index) in self.sv means a higher function value.
        """
        return self._vertex_rank.get(v1, -1) < self._vertex_rank.get(v2, -1)

    # --- DSU (Disjoint Set Union) Implementation for 'nodes' ---

    def find(self, i: int) -> int:
        """DSU Find operation with path compression."""
        # Ensure the vertex ID is valid and not already a root
        if i >= len(self.parent) or self.parent[i] == i:
            return i
        
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, root1: int, root2: int) -> bool:
        """
        DSU Union operation (Merge). Merges the component of 'root1' into the component of 'root2'.
        """
        root_v = self.find(root2)
        if root_v != root1:
            self.parent[root1] = root_v
            return True
        return False

    # --- Ported Core Functions ---

    def process_vertex(self, v: int):
        """
        Port of the C++ 'void MergeTree::processVertex(int64_t v)' function.
        Focuses on component finding, merging, and tree edge creation.
        """
        star_neighbors = self.get_star(v)
        
        if not star_neighbors: 
            return

        set_of_components = set()
        
        # Identify unique components connected by an upper link (f(v) < f(tin))
        for tin in star_neighbors:
            if self.is_upper_link(v, tin):
                comp = self.find(tin)
                set_of_components.add(comp)

        # Case 1: No upper links
        if not set_of_components: 
            # Establish v as the Critical Point (CP) for its component.
            comp_v = self.find(v)
            self.cp_map[comp_v] = v
        
        # Case 2: One or more upper links 
        else:
            # Merge all connected upper components into the component of v
            for comp in set_of_components:
                # Find the critical point (CP) of the component being merged (the child node)
                to_vertex = self.cp_map.get(comp)
                
                # If a valid CP exists for the component, create a tree edge
                if to_vertex is not None:
                    from_vertex = v
                    # Create the tree edge: (v, to_vertex)
                    self.prev[to_vertex] = from_vertex 
                
                # Perform the DSU merge operation: comp's root becomes a child of v's root
                self.union(comp, v)
            
            # Update the component map: v is the new CP for the merged component
            comp_v_new = self.find(v)
            self.cp_map[comp_v_new] = v

    def compute_join_tree(self):
        """
        Port of the C++ 'void MergeTree::computeJoinTree()' function.
        Processes vertices in reverse sorted order (high function value to low function value).
        """
        print("Computing Join Tree...")
        
        ct = 0
        
        # Process vertices in reverse sorted order
        for i in range(self.no_vertices - 1, -1, -1):

            ct += 1
            
            v = self.sv[i]
            self.process_vertex(v)
        
        # Finalizing root (last part of original C++ function)
        in_idx = 0

        self.new_root = in_idx
        print(f"Setting new_root index = {self.new_root}")

def neighbours_from_adjacency(input_edge_list, input_sorted_vertices):
    """Work out input_neighbor_map from input_edge_list (Adjacency List Construction) ---
    """
    all_vertices = set(input_sorted_vertices) 
    # Use sets to automatically handle bidirectional edges and avoid duplicates
    input_neighbor_map_sets = {v: set() for v in all_vertices}
    
    for u, v in input_edge_list:
        # Check if vertices exist in our critical point set before adding neighbors
        if u in input_neighbor_map_sets and v in input_neighbor_map_sets:
            # Add neighbors symmetrically (u is neighbor of v, v is neighbor of u)
            input_neighbor_map_sets[u].add(v)
            input_neighbor_map_sets[v].add(u)
            
    # Convert sets back to lists for the final map structure
    input_neighbor_map = {k: list(v) for k, v in input_neighbor_map_sets.items()}
    return input_neighbor_map

# --- Example Usage for Testing ---
if __name__ == "__main__":
    
    # Define a graph where all 5 vertices are critical points.
    # Sorted Vertices (sv): [0, 1, 3, 2, 4] (from min value 0 to max value 4)
    
    input_sorted_vertices = [0, 1, 3, 2, 4]
    input_edge_list = [
        (0, 1), 
        (1, 2), 
        (2, 3), 
        (2, 4)
    ]
    input_neighbor_map = neighbours_from_adjacency(input_edge_list, input_sorted_vertices)
    
    # All points are assumed to be classified as critical points
    input_critical_points_map = {
        0: MINIMUM, # Global Minimum
        1: MINIMUM, # Local Minimum (f(1)=1)
        3: SADDLE,  # Saddle (f(3)=2) - Note: Updated to be a saddle for a more interesting example
        2: MAXIMUM, # Local Maximum (f(2)=3)
        4: MAXIMUM  # Global Maximum (f(4)=4)
    }
    
    print("\n--- Initializing Merge Tree Processor ---")
    print(f"Sorted Vertices (Min -> Max Value): {input_sorted_vertices}")
    
    try:
        tree_processor = MergeTree(
            sorted_vertices=input_sorted_vertices,
            neighbor_map=input_neighbor_map,
            critical_points_map=input_critical_points_map
        )
        
        print("\n--- Running compute_join_tree ---")
        tree_processor.compute_join_tree()
        
        print("\n--- Final State ---")
        print(f"new_root: {tree_processor.new_root}")
        print("--- Critical Points (ID: Type) ---")
        # Print all critical points from the input map
        print({v: ('SADDLE' if t == SADDLE else 'MAX' if t == MAXIMUM else 'MIN') 
               for v, t in tree_processor.critical_pts.items()})
        print("--- Tree Edges (Parent Pointers) ---")
        # prev maps component's CP (child) -> merging vertex (parent)
        print({f"{k} -> {v}": f"Edge from CP {k} to {v}" for k, v in tree_processor.prev.items()})

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")