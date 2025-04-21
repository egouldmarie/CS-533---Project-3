import cProfile
import pstats
import time
import os
from hopkins_stat import hopkins_stat, read_data
from Grid import Grid
from BoundingBox import BoundingBox

# Create output directory
OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test configuration
DATASET = "DS-1"
DEFAULT_N = 2000
DEFAULT_M = 50
DEFAULT_B = 20

def get_dataset_filename(dataset, n):
    if n == 5000:
        return f"{dataset}/dataset-{dataset[-1]}-1.csv"
    else:
        return f"{dataset}/dataset-{dataset[-1]}-1-{n}.csv"

def initialize_grid(dataset_file):
    dataset = read_data(dataset_file)
    bbox = BoundingBox(x_min=-10000, x_max=10000, y_min=-10000, y_max=10000)
    grid = Grid(dataset, bbox=bbox, partition_width=1000)
    return grid, len(dataset)

def profile_hopkins_stat(grid, m, runs=1):
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()
    for _ in range(runs):
        hopkins_stat(grid, m)
    elapsed_time = (time.time() - start_time) / runs
    
    profiler.disable()

    stats = pstats.Stats(profiler)
    return stats, elapsed_time

def test_single_run_profiling(dataset=DATASET, n=DEFAULT_N, m=DEFAULT_M):
    print(f"\nDetailed performance analysis of a single run (dataset={dataset}, n={n}, m={m})")
    
    dataset_file = get_dataset_filename(dataset, n)
    grid, actual_n = initialize_grid(dataset_file)
    
    # Perform performance analysis
    stats, elapsed_time = profile_hopkins_stat(grid, m, runs=3)

    stats_file = f"{OUTPUT_DIR}/profile_stats_{dataset}_n{n}_m{m}.txt"
    with open(stats_file, 'w') as f:

        stats.sort_stats('cumulative').stream = f
        stats.print_stats(20)
    
    print(f"  Single run time: {elapsed_time:.4f} seconds")
    print(f"  Detailed performance statistics saved to {stats_file}")

    print("\nMost time-consuming functions:")
    stats.sort_stats('cumulative').print_stats(5)
    
    return stats, elapsed_time

def test_system_load_impact(dataset=DATASET, n=DEFAULT_N, m=DEFAULT_M, b=DEFAULT_B):
    """Test the impact of system load on performance"""
    print(f"\nTesting the impact of system load on performance (dataset={dataset}, n={n}, m={m}, b={b})")
    
    dataset_file = get_dataset_filename(dataset, n)
    grid, actual_n = initialize_grid(dataset_file)

    print("  Running under light load...")
    light_start_time = time.time()
    light_cpu_start = time.process_time()
    
    _ = [hopkins_stat(grid, m) for _ in range(b)]
    
    light_cpu_time = time.process_time() - light_cpu_start
    light_wall_time = time.time() - light_start_time

    print("  Running under heavy load...")

    heavy_matrices = []
    for _ in range(2):
        matrix = []
        for i in range(1000):
            row = []
            for j in range(1000):
                row.append(i*j)
            matrix.append(row)
        heavy_matrices.append(matrix)
        
    heavy_start_time = time.time()
    heavy_cpu_start = time.process_time()
    
    _ = [hopkins_stat(grid, m) for _ in range(b)]
    
    heavy_cpu_time = time.process_time() - heavy_cpu_start
    heavy_wall_time = time.time() - heavy_start_time
    
    print(f"  Light load: CPU time={light_cpu_time:.4f}s, Wall time={light_wall_time:.4f}s")
    print(f"  Heavy load: CPU time={heavy_cpu_time:.4f}s, Wall time={heavy_wall_time:.4f}s")
    print(f"  CPU time ratio (heavy/light): {heavy_cpu_time/light_cpu_time:.2f}")
    print(f"  Wall time ratio (heavy/light): {heavy_wall_time/light_wall_time:.2f}")
    
    return {
        'light_cpu_time': light_cpu_time,
        'light_wall_time': light_wall_time,
        'heavy_cpu_time': heavy_cpu_time,
        'heavy_wall_time': heavy_wall_time
    }

def quick_varying_test():
    """Quick test of the impact of different parameter values"""
    print(f"\nQuick test of the impact of different parameter values (dataset={DATASET})")
    
    dataset_file = get_dataset_filename(DATASET, DEFAULT_N)
    grid, actual_n = initialize_grid(dataset_file)
    
    # Test different m values
    m_values = [10, 50, 100]
    m_results = []
    for m in m_values:
        print(f"  Testing m={m}...")
        start_time = time.time()
        _ = hopkins_stat(grid, m)
        elapsed_time = time.time() - start_time
        m_results.append((m, elapsed_time))
    
    # Test different B values (bootstrap iterations)
    b_values = [5, 10, 20]
    b_results = []
    for b in b_values:
        print(f"  Testing B={b}...")
        start_time = time.time()
        _ = [hopkins_stat(grid, DEFAULT_M) for _ in range(b)]
        elapsed_time = time.time() - start_time
        b_results.append((b, elapsed_time))
    
    # Print results
    print("\nImpact of m value on performance:")
    for m, time_elapsed in m_results:
        print(f"  m={m}: {time_elapsed:.4f} seconds")
    
    print("\nImpact of B value on performance:")
    for b, time_elapsed in b_results:
        print(f"  B={b}: {time_elapsed:.4f} seconds")
    
    return m_results, b_results

if __name__ == "__main__":
    import platform
    print("System Information:")
    print(f"  Operating System: {platform.system()} {platform.version()}")
    print(f"  Python Version: {platform.python_version()}")
    print(f"  Processor: {platform.processor()}")
    
    # Run quick tests
    test_single_run_profiling()
    quick_varying_test()
    test_system_load_impact() 