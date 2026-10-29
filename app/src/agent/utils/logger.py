import time
from functools import wraps
from typing import Callable, Any

def track_node_metrics(node_name: str) -> Callable:
    """
    A decorator that logs execution time and token usage for any LangGraph node.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: dict, *args, **kwargs) -> dict:
            start_time = time.time()
            print(f"\n🚀 [Node: {node_name}] Execution started...")
            
            # Execute the node function
            result = func(state, *args, **kwargs)
            
            elapsed_time = time.time() - start_time
            print(f"⏱ [Node: {node_name}] Finished in {elapsed_time:.4f}s")
            
            return result
        return wrapper
    return decorator