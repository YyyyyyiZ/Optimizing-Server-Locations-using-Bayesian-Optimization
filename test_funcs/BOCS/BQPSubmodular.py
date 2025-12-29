import numpy as np
import igraph as ig


class BQPSubmodular:
    def __init__(self, n_vars, fix_num=None):
        self.n_vars = n_vars
        self.fix_num = fix_num
        self.gamma = np.ones(n_vars)  # Initialize relaxation parameters
        self.A_pos = None  # Add A_pos as member variable

    def solve(self, b, A, max_iter=10):
        """Main solver function: combines submodular relaxation with cardinality constrained graph cuts"""
        # Decompose A = A_neg + A_pos
        self.A_pos = np.where(A > 0, A, 0)  # Save as member variable
        A_neg = np.where(A < 0, A, 0)

        best_x, best_obj = None, np.inf

        for _ in range(max_iter):
            # Relax non-submodular part A^+
            relaxed_linear = self._relax_A_pos(self.A_pos)

            # Construct graph and solve
            if self.fix_num:
                x, obj = self._solve_graph_cut_constraint(b + relaxed_linear, A_neg)
            else:
                x, obj = self._solve_graph_cut(b + relaxed_linear, A_neg)

            # Update relaxation parameter gamma (based on current solution x)
            self._update_gamma(x, self.A_pos)

            if obj < best_obj:
                best_x, best_obj = x, obj

        return best_x, best_obj

    def _relax_A_pos(self, A_pos):
        """Linear relaxation of non-submodular part A^+: x_i x_j <= gamma_i x_i + gamma_j x_j"""
        return np.sum(A_pos * self.gamma.reshape(-1, 1), axis=0) + \
            np.sum(A_pos * self.gamma.reshape(1, -1), axis=1)

    def _solve_graph_cut_constraint(self, b, A_neg):
        """Solve graph cut for submodular part A^- with cardinality constraint"""
        # Create directed graph (igraph vertex indices start from 0)
        g = ig.Graph(directed=True)

        # Add vertices: source (s=0), sink (t=1), variable nodes (u_i=2~n+1, v_i=n+2~2n+1)
        n = self.n_vars
        g.add_vertices(2 + 2 * n)  # Total vertices = 2 (s,t) + 2n (u_i,v_i)

        # Large constant to enforce cardinality constraint
        LARGE_VALUE = 1e6 * np.max(np.abs(b))

        # Store edge capacities (igraph uses edge weight property)
        edge_capacities = []

        # Add linear term edges (s -> u_i and v_i -> t)
        for i in range(n):
            # s -> u_i (forward edge, capacity = b_i^+)
            if b[i] > 0:
                g.add_edge(0, 2 + i)  # s=0, u_i=2+i
                edge_capacities.append(b[i])

            # v_i -> t (backward edge, capacity = b_i^- + LARGE_VALUE)
            cap = max(0, -b[i]) + LARGE_VALUE
            g.add_edge(2 + n + i, 1)  # v_i=2+n+i, t=1
            edge_capacities.append(cap)

        # Add quadratic term edges (u_i -> v_j, only for A^-)
        for i in range(n):
            for j in range(i + 1, n):
                if A_neg[i, j] != 0:
                    g.add_edge(2 + i, 2 + n + j)  # u_i -> v_j
                    edge_capacities.append(-A_neg[i, j])

        # Solve max-flow/min-cut (using push-relabel algorithm)
        flow_value = g.maxflow(0, 1, edge_capacities)

        # Method 1: Directly get partition marks (need to confirm attribute name)
        if "value" in g.vs.attributes():
            partition = g.vs["value"]
        else:
            # Method 2: Get partition via mincut
            cut = g.mincut(0, 1, edge_capacities)
            partition = [0 if v.index in cut.partition[0] else 1 for v in g.vs]

        # Extract solution: x_i=1 iff u_i is on source side and v_i is on sink side
        x = np.zeros(n, dtype=int)
        for i in range(n):
            if partition[2 + i] == 0 and partition[2 + n + i] == 1:  # u_i on source side and v_i on sink side
                x[i] = 1

        # Verify cardinality constraint
        if sum(x) != self.fix_num:
            # print("Submodular relaxation failed")
            # If cardinality constraint not satisfied, force select top fix_num variables with largest b_i
            sorted_indices = np.argsort(b)[::-1]
            x = np.zeros(n, dtype=int)
            x[sorted_indices[:self.fix_num]] = 1

        # Calculate objective value (only for current relaxed problem)
        obj = x @ (A_neg + self.A_pos) @ x + b @ x  # Use member variable self.A_pos
        return x, obj

    def _solve_graph_cut(self, b, A_neg):
        """Solve graph cut for submodular part A^- without cardinality constraint"""
        # Create directed graph (igraph vertex indices start from 0)
        g = ig.Graph(directed=True)

        # Add vertices: source (s=0), sink (t=1), variable nodes (u_i=2~n+1)
        n = self.n_vars
        g.add_vertices(2 + n)  # Total vertices = 2 (s,t) + n (u_i)

        # Store edge capacities (igraph uses edge weight property)
        edge_capacities = []

        # Add linear term edges (s -> u_i and u_i -> t)
        for i in range(n):
            # s -> u_i (forward edge, capacity = b_i^+)
            if b[i] > 0:
                g.add_edge(0, 2 + i)  # s=0, u_i=2+i
                edge_capacities.append(b[i])
            # u_i -> t (backward edge, capacity = b_i^-)
            else:
                g.add_edge(2 + i, 1)  # u_i=2+i, t=1
                edge_capacities.append(-b[i])

        # Add quadratic term edges (u_i -> u_j, only for A^-)
        for i in range(n):
            for j in range(i + 1, n):
                if A_neg[i, j] != 0:
                    g.add_edge(2 + i, 2 + j)  # u_i -> u_j
                    edge_capacities.append(-A_neg[i, j])
                    g.add_edge(2 + j, 1)  # u_j -> t
                    edge_capacities.append(-A_neg[i, j])

        # Solve max-flow/min-cut (using push-relabel algorithm)
        flow_value = g.maxflow(0, 1, edge_capacities)

        # Method 1: Directly get partition marks (need to confirm attribute name)
        if "value" in g.vs.attributes():
            partition = g.vs["value"]
        else:
            # Method 2: Get partition via mincut
            cut = g.mincut(0, 1, edge_capacities)
            partition = [0 if v.index in cut.partition[0] else 1 for v in g.vs]

        # Extract solution: 1 if on target side, 0 otherwise
        x = np.zeros(n, dtype=int)
        for i in range(n):
            if partition[2 + i] == 1:
                x[i] = 1

        # Calculate objective value (only for current relaxed problem)
        obj = x @ (A_neg + self.A_pos) @ x + b @ x
        return x, obj

    def _update_gamma(self, x, A_pos):
        """Update relaxation parameter gamma based on current solution"""
        grad = np.sum(A_pos * (1 - x.reshape(-1, 1)), axis=1)  # Gradient direction
        self.gamma -= 0.1 * grad  # Simple gradient descent
        self.gamma = np.clip(self.gamma, 0, 1)  # Project to [0,1]


if __name__ == '__main__':
    n_vars = 10
    fix_num = 3  # Must select exactly 3 variables as 1
    b = np.random.randn(n_vars)  # Linear terms
    A = np.random.randn(n_vars, n_vars)
    A = (A + A.T) / 2  # Symmetrize

    solver = BQPSubmodular(n_vars, fix_num)
    x_opt, obj_opt = solver.solve(b, A)

    print(f"Optimal solution: {x_opt}")
    print(f"Optimal Value: {obj_opt:.4f}")