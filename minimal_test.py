import time
import random
import numpy as np
from hopkins_stat import hopkins_stat, read_data
from Grid import Grid
from BoundingBox import BoundingBox

# Test configuration
DATASET = "DS-1"
DEFAULT_N = 2000  # Use a smaller dataset
DEFAULT_M = 50    # Use fewer random points
DEFAULT_B = 5     # Use very few bootstrap iterations for quick testing

def get_dataset_filename(dataset, n):
    """Get the filename for a dataset of specified size"""
    if n == 5000:
        return f"{dataset}/dataset-{dataset[-1]}-1.csv"
    else:
        return f"{dataset}/dataset-{dataset[-1]}-1-{n}.csv"

def initialize_grid(dataset_file):
    """Initialize Grid object, return Grid and dataset size"""
    print(f"Reading dataset: {dataset_file}")
    dataset = read_data(dataset_file)
    bbox = BoundingBox(x_min=-10000, x_max=10000, y_min=-10000, y_max=10000)
    grid = Grid(dataset, bbox=bbox, partition_width=1000)
    return grid, len(dataset)

def test_hopkins_time(grid, m):
    """Measure time for a single hopkins_stat call"""
    start_time = time.time()
    result = hopkins_stat(grid, m)
    elapsed_time = time.time() - start_time
    return elapsed_time, result

def basic_performance_test():
    """Basic performance test"""
    print("\nBasic Performance Test")
    
    # Load dataset
    dataset_file = get_dataset_filename(DATASET, DEFAULT_N)
    grid, n = initialize_grid(dataset_file)
    print(f"Dataset size: {n} points")
    
    # Test different m values performance
    m_values = [10, 50, 100]
    print("\nTesting different m values:")
    for m in m_values:
        total_time = 0
        iterations = 5
        for i in range(iterations):
            elapsed_time, _ = test_hopkins_time(grid, m)
            total_time += elapsed_time
        avg_time = total_time / iterations
        print(f"  m={m}: Average time={avg_time:.6f} seconds")
    
    # Test bootstrap process (different number of iterations)
    b_values = [5, 10, 20]
    print("\nTesting different B values:")
    for b in b_values:
        start_time = time.time()
        results = [hopkins_stat(grid, DEFAULT_M) for _ in range(b)]
        elapsed_time = time.time() - start_time
        print(f"  B={b}: Total time={elapsed_time:.6f} seconds, Average per call={elapsed_time/b:.6f} seconds")

def test_n_impact():
    """Test the impact of dataset size on performance"""
    print("\nImpact of dataset size on performance")
    
    n_values = [2000, 2900, 3800]  # Modified to use actual existing files
    results = []
    
    for n in n_values:
        dataset_file = get_dataset_filename(DATASET, n)
        grid, actual_n = initialize_grid(dataset_file)
        
        # Average multiple runs for more stable results
        total_time = 0
        iterations = 3
        for i in range(iterations):
            elapsed_time, _ = test_hopkins_time(grid, DEFAULT_M)
            total_time += elapsed_time
        avg_time = total_time / iterations
        
        results.append((n, avg_time))
        print(f"  n={n}: Average time={avg_time:.6f} seconds")
    
    return results

if __name__ == "__main__":
    import platform
    print("System Information:")
    print(f"  Operating System: {platform.system()} {platform.version()}")
    print(f"  Python Version: {platform.python_version()}")
    print(f"  Processor: {platform.processor()}")
    
    # Run tests
    basic_performance_test()
    test_n_impact() 