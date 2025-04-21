import cProfile
import pstats
import time
import numpy as np
from hopkins_stat import hopkins_stat, read_data
from Grid import Grid
from BoundingBox import BoundingBox

# Test configuration
DATASET = "DS-1"
DEFAULT_N = 2000  
DEFAULT_M = 50    
DEFAULT_B = 10    

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

def profile_with_cProfile(grid, m, iterations=1):
    """Profile hopkins_stat function performance"""
    print(f"\ncProfile performance analysis (m={m}, iterations={iterations})")

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(iterations):
        hopkins_stat(grid, m)
    
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumulative')
    print("\nFunction call statistics (sorted by cumulative time):")
    stats.print_stats(10)  # Only show top 10 functions
    
    print("\nSorted by number of calls:")
    stats = pstats.Stats(profiler).sort_stats('calls')
    stats.print_stats(10)
    
    return stats

def test_parameter_impact():
    """Test the impact of different parameters on performance"""
    # Load dataset
    dataset_file = get_dataset_filename(DATASET, DEFAULT_N)
    grid, n = initialize_grid(dataset_file)
    
    # Test different m values
    print("\nTesting the performance impact of different m values:")
    m_values = [10, 50, 100]
    m_results = []
    
    for m in m_values:
        start_time = time.time()
        for _ in range(5):  # Run multiple times for more stable results
            hopkins_stat(grid, m)
        elapsed_time = (time.time() - start_time) / 5
        m_results.append((m, elapsed_time))
        print(f"  m={m}: Average time={elapsed_time:.6f} seconds")

    print("\nTesting the performance impact of different dataset sizes:")
    n_values = [2000, 2900, 3800]
    n_results = []
    
    for n in n_values:
        dataset_file = get_dataset_filename(DATASET, n)
        grid, actual_n = initialize_grid(dataset_file)
        
        start_time = time.time()
        for _ in range(5):  # Run multiple times for more stable results
            hopkins_stat(grid, DEFAULT_M)
        elapsed_time = (time.time() - start_time) / 5
        n_results.append((n, elapsed_time))
        print(f"  n={n}: Average time={elapsed_time:.6f} seconds")
    
    return m_results, n_results

if __name__ == "__main__":
    import platform
    print("System Information:")
    print(f"  Operating System: {platform.system()} {platform.version()}")
    print(f"  Python Version: {platform.python_version()}")
    print(f"  Processor: {platform.processor()}")
    dataset_file = get_dataset_filename(DATASET, DEFAULT_N)
    grid, n = initialize_grid(dataset_file)

    profile_with_cProfile(grid, DEFAULT_M, iterations=1)

    test_parameter_impact() 