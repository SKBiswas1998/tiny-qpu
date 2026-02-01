"""
QAOA (Quantum Approximate Optimization Algorithm) for MaxCut.

Solves the MaxCut problem: partition a graph into two sets to 
maximize the number of edges between them.

Usage:
    from tiny_qpu.apps import QAOA
    
    # Define graph as edge list
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    
    # Solve
    qaoa = QAOA(edges, p=1)
    result = qaoa.optimize()
    
    print(f"Best cut: {result.bitstring}")
    print(f"Cut value: {result.cost}")
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
from ..core import Circuit


@dataclass
class QAOAResult:
    """Result of QAOA optimization."""
    bitstring: str           # Best solution found
    cost: float              # Cost (cut value) of best solution
    optimal_params: Tuple[np.ndarray, np.ndarray]  # (gamma, beta)
    history: List[float]     # Optimization history
    counts: Dict[str, int]   # Final measurement counts
    
    def cut_value(self) -> int:
        """Return the cut value as integer."""
        return int(round(self.cost))


class QAOA:
    """
    Quantum Approximate Optimization Algorithm for MaxCut.
    
    The MaxCut problem: Given a graph G=(V,E), find a partition of 
    vertices into two sets that maximizes edges crossing the partition.
    
    QAOA encodes the problem into a quantum circuit with p layers,
    each controlled by parameters (γ, β). Classical optimization
    finds the best parameters.
    
    Example:
        >>> # Square graph with diagonal
        >>> edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
        >>> qaoa = QAOA(edges, p=2)
        >>> result = qaoa.optimize()
        >>> print(f"Best partition: {result.bitstring}")
        >>> print(f"Edges cut: {result.cut_value()}")
    
    Args:
        edges: List of (i, j) tuples representing graph edges
        p: Number of QAOA layers (higher = better but slower)
        weights: Optional edge weights (default: all 1)
    """
    
    def __init__(self, edges: List[Tuple[int, int]], p: int = 1,
                 weights: Optional[List[float]] = None):
        self.edges = edges
        self.p = p
        self.weights = weights or [1.0] * len(edges)
        
        # Determine number of qubits (nodes in graph)
        self.num_qubits = max(max(e) for e in edges) + 1
        
        # Optimization history
        self._history: List[float] = []
    
    def _cost_operator(self, qc: Circuit, gamma: float) -> None:
        """
        Apply cost unitary: exp(-i γ C) where C is the MaxCut Hamiltonian.
        
        For MaxCut: C = Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2
        This translates to RZZ gates on each edge.
        """
        for (i, j), w in zip(self.edges, self.weights):
            # exp(-i γ w (1 - ZZ)/2) = exp(-i γ w/2) exp(i γ w ZZ/2)
            # The global phase doesn't matter, so we just apply RZZ
            qc.rzz(gamma * w, i, j)
    
    def _mixer_operator(self, qc: Circuit, beta: float) -> None:
        """
        Apply mixer unitary: exp(-i β B) where B = Σ_i X_i.
        
        This is just Rx(2β) on each qubit.
        """
        for i in range(self.num_qubits):
            qc.rx(2 * beta, i)
    
    def _build_circuit(self, gamma: np.ndarray, beta: np.ndarray) -> Circuit:
        """Build the QAOA circuit with given parameters."""
        qc = Circuit(self.num_qubits)
        
        # Initial state: uniform superposition |+⟩^n
        for i in range(self.num_qubits):
            qc.h(i)
        
        # Apply p layers
        for layer in range(self.p):
            self._cost_operator(qc, gamma[layer])
            self._mixer_operator(qc, beta[layer])
        
        # Measure all qubits
        qc.measure_all()
        
        return qc
    
    def _compute_cut_value(self, bitstring: str) -> float:
        """Compute cut value for a given bitstring."""
        cut = 0.0
        for (i, j), w in zip(self.edges, self.weights):
            # Edge is cut if bits differ
            if bitstring[-(i+1)] != bitstring[-(j+1)]:
                cut += w
        return cut
    
    def _expectation(self, params: np.ndarray, shots: int = 1024) -> float:
        """Compute expected cost for given parameters."""
        gamma = params[:self.p]
        beta = params[self.p:]
        
        qc = self._build_circuit(gamma, beta)
        result = qc.run(shots=shots)
        
        # Compute expectation value
        total_cost = 0.0
        for bitstring, count in result.counts.items():
            total_cost += self._compute_cut_value(bitstring) * count
        
        expected = total_cost / shots
        self._history.append(expected)
        
        # Return negative because we minimize (want to maximize cut)
        return -expected
    
    def optimize(self, shots: int = 1024, method: str = 'COBYLA',
                 maxiter: int = 100, seed: Optional[int] = None,
                 initial_params: Optional[np.ndarray] = None) -> QAOAResult:
        """
        Run QAOA optimization.
        
        Args:
            shots: Measurement shots per circuit evaluation
            method: Scipy optimization method ('COBYLA', 'SLSQP', 'Nelder-Mead')
            maxiter: Maximum optimizer iterations
            seed: Random seed for reproducibility
            initial_params: Starting parameters (default: random)
            
        Returns:
            QAOAResult with best solution found
        """
        if seed is not None:
            np.random.seed(seed)
        
        self._history = []
        
        # Initial parameters
        if initial_params is None:
            gamma0 = np.random.uniform(0, 2*np.pi, self.p)
            beta0 = np.random.uniform(0, np.pi, self.p)
            x0 = np.concatenate([gamma0, beta0])
        else:
            x0 = initial_params
        
        # Optimize
        result = minimize(
            lambda x: self._expectation(x, shots),
            x0,
            method=method,
            options={'maxiter': maxiter}
        )
        
        optimal_gamma = result.x[:self.p]
        optimal_beta = result.x[self.p:]
        
        # Run final circuit with optimal parameters
        qc = self._build_circuit(optimal_gamma, optimal_beta)
        final_result = qc.run(shots=shots * 10, seed=seed)  # More shots for final
        
        # Find best bitstring
        best_bitstring = max(final_result.counts.keys(),
                            key=lambda x: self._compute_cut_value(x))
        best_cost = self._compute_cut_value(best_bitstring)
        
        return QAOAResult(
            bitstring=best_bitstring,
            cost=best_cost,
            optimal_params=(optimal_gamma, optimal_beta),
            history=self._history,
            counts=final_result.counts
        )
    
    def landscape(self, gamma_range: Tuple[float, float] = (0, np.pi),
                  beta_range: Tuple[float, float] = (0, np.pi/2),
                  resolution: int = 20, shots: int = 256) -> np.ndarray:
        """
        Compute parameter landscape for p=1 QAOA.
        
        Useful for visualization and understanding the optimization.
        
        Returns:
            2D array of expected costs for (gamma, beta) grid
        """
        if self.p != 1:
            raise ValueError("Landscape visualization only supported for p=1")
        
        gammas = np.linspace(gamma_range[0], gamma_range[1], resolution)
        betas = np.linspace(beta_range[0], beta_range[1], resolution)
        
        landscape = np.zeros((resolution, resolution))
        
        for i, gamma in enumerate(gammas):
            for j, beta in enumerate(betas):
                params = np.array([gamma, beta])
                landscape[i, j] = -self._expectation(params, shots)
        
        self._history = []  # Clear history from landscape
        return landscape
    
    @staticmethod
    def from_adjacency_matrix(adj_matrix: np.ndarray, p: int = 1) -> 'QAOA':
        """Create QAOA instance from adjacency matrix."""
        edges = []
        weights = []
        n = adj_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] != 0:
                    edges.append((i, j))
                    weights.append(float(adj_matrix[i, j]))
        
        return QAOA(edges, p=p, weights=weights)
    
    @staticmethod
    def random_graph(num_nodes: int, edge_prob: float = 0.5,
                     p: int = 1, seed: Optional[int] = None) -> 'QAOA':
        """Create QAOA instance for a random Erdos-Renyi graph."""
        if seed is not None:
            np.random.seed(seed)
        
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < edge_prob:
                    edges.append((i, j))
        
        if not edges:
            # Ensure at least one edge
            edges = [(0, 1)]
        
        return QAOA(edges, p=p)


def solve_maxcut(edges: List[Tuple[int, int]], p: int = 1,
                 shots: int = 1024, seed: Optional[int] = None) -> QAOAResult:
    """
    Convenience function to solve MaxCut.
    
    Example:
        >>> edges = [(0,1), (1,2), (2,0)]  # Triangle
        >>> result = solve_maxcut(edges, p=2)
        >>> print(f"Max cut: {result.cut_value()}")
    """
    qaoa = QAOA(edges, p=p)
    return qaoa.optimize(shots=shots, seed=seed)
