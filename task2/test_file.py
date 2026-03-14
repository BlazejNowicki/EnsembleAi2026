import math

def calculate_metrics(values: list[float]) -> dict:
    """Tests high-level function parsing with type hinting."""
    
    def clean(v):
        """Nested helper to test depth-aware chunking."""
        return v if v is not None else 0.0

    if not values:
        return {"error": "empty"}

    processed = [clean(x) for x in values]
    
    # Testing lambda parsing within a function
    avg_func = lambda x: sum(x) / len(x)
    
    return {
        "mean": avg_func(processed),
        "max": max(processed)
    }

async def stream_data(source_url: str):
    """Tests async method parsing."""
    try:
        # Simulate an async operation
        return f"Data from {source_url}"
    except Exception as e:
        return f"Failure: {str(e)}"

def recursive_factorial(n: int) -> int:
    """Tests if the chunker handles self-referential function names."""
    if n <= 1:
        return 1
    return n * recursive_factorial(n - 1)

def _utility_private():
    """Testing filtering of underscore-prefixed methods."""
    return True