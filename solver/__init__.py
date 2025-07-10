# solver/__init__.py
from .base_solver import BaseSolver
from .red_cnn_solver import RedCNNSolver 

def get_solver(name: str) -> type[BaseSolver]:
    solvers = {
        'red_cnn': RedCNNSolver
    }

    name = name.lower()
    if name not in solvers:
        raise ValueError(f"Invalid solver name '{name}'. Available: {list(solvers.keys())}")

    return solvers[name]