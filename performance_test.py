import cProfile
import pstats
import time
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from hopkins_stat import hopkins_stat, read_data
from Grid import Grid
from BoundingBox import BoundingBox

# Performance test configuration
OUTPUT_DIR = "performance_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed parameters
DATASETS = ["DS-1", "DS-2", "DS-3", "DS-4"]
DEFAULT_N = 5000  # Default dataset size
DEFAULT_M = 100   # Default number of random points
DEFAULT_B = 1000  # Default number of bootstrap iterations

# Parameter ranges
B_RANGE = [1000, 3000, 5000, 7000, 9000, 11000]
M_RANGE = [100, 300, 500, 700, 900, 1100]
N_RANGE = [500, 1000, 2000, 3000, 4000, 5000]

def get_dataset_filename(dataset, n):
    """Get the filename for a dataset of specified size"""
    if n == 5000:
        return f"{dataset}/dataset-{dataset[-1]}-1.csv"
    else:
        return f"{dataset}/dataset-{dataset[-1]}-1-{n}.csv"

def initialize_grid(dataset_file):
    """Initialize Grid object, return Grid and dataset size"""
    dataset = read_data(dataset_file)
    bbox = BoundingBox(x_min=-10000, x_max=10000, y_min=-10000, y_max=10000)
    grid = Grid(dataset, bbox=bbox, partition_width=1000)
    return grid, len(dataset)

def profile_hopkins_stat(grid, m, runs=1):
    """Profile hopkins_stat function performance using cProfile"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run multiple times to get more stable results
    start_time = time.time()
    for _ in range(runs):
        hopkins_stat(grid, m)
    elapsed_time = (time.time() - start_time) / runs
    
    profiler.disable()

    stats = pstats.Stats(profiler)
    return stats, elapsed_time

def run_bootstrap_test(grid, m, b):
    """Measure time to run B bootstrap iterations"""
    start_time = time.time()
    results = [hopkins_stat(grid, m) for _ in range(b)]
    elapsed_time = time.time() - start_time
    return elapsed_time, results

def test_varying_b(dataset="DS-1", n=DEFAULT_N, m=DEFAULT_M):
    """Test the performance impact of different B values"""
    print(f"\nTesting the performance impact of different B values (dataset={dataset}, n={n}, m={m})")
    results = []
    
    dataset_file = get_dataset_filename(dataset, n)
    grid, actual_n = initialize_grid(dataset_file)
    
    for b in B_RANGE:
        print(f"  Running B={b}...")
        elapsed_time, _ = run_bootstrap_test(grid, m, b)
        results.append({
            'dataset': dataset,
            'n': actual_n,
            'm': m,
            'B': b,
            'time': elapsed_time
        })
    
    # Save results
    result_file = f"{OUTPUT_DIR}/varying_b_{dataset}_n{n}_m{m}.csv"
    with open(result_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'n', 'm', 'B', 'time'])
        writer.writeheader()
        writer.writerows(results)
    
    return results

def test_varying_m(dataset="DS-1", n=DEFAULT_N, b=DEFAULT_B):
    """Test the performance impact of different m values"""
    print(f"\nTesting the performance impact of different m values (dataset={dataset}, n={n}, b={b})")
    results = []
    
    dataset_file = get_dataset_filename(dataset, n)
    grid, actual_n = initialize_grid(dataset_file)
    
    for m in M_RANGE:
        print(f"  Running m={m}...")
        elapsed_time, _ = run_bootstrap_test(grid, m, b)
        results.append({
            'dataset': dataset,
            'n': actual_n,
            'm': m,
            'B': b,
            'time': elapsed_time
        })

    result_file = f"{OUTPUT_DIR}/varying_m_{dataset}_n{n}_b{b}.csv"
    with open(result_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'n', 'm', 'B', 'time'])
        writer.writeheader()
        writer.writerows(results)
    
    return results

def test_varying_n(dataset="DS-1", m=DEFAULT_M, b=DEFAULT_B):
    """Test the performance impact of different n values"""
    print(f"\nTesting the performance impact of different n values (dataset={dataset}, m={m}, b={b})")
    results = []
    
    for n in N_RANGE:
        print(f"  Running n={n}...")
        dataset_file = get_dataset_filename(dataset, n)
        grid, actual_n = initialize_grid(dataset_file)
        
        elapsed_time, _ = run_bootstrap_test(grid, m, b)
        results.append({
            'dataset': dataset,
            'n': actual_n,
            'm': m,
            'B': b,
            'time': elapsed_time
        })
    
    # Save results
    result_file = f"{OUTPUT_DIR}/varying_n_{dataset}_m{m}_b{b}.csv"
    with open(result_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'n', 'm', 'B', 'time'])
        writer.writeheader()
        writer.writerows(results)
    
    return results

def test_single_run_profiling(dataset="DS-1", n=DEFAULT_N, m=DEFAULT_M):
    """Detailed performance analysis of a single run"""
    print(f"\nDetailed performance analysis of a single run (dataset={dataset}, n={n}, m={m})")
    
    dataset_file = get_dataset_filename(dataset, n)
    grid, actual_n = initialize_grid(dataset_file)
    
    # Perform performance analysis
    stats, elapsed_time = profile_hopkins_stat(grid, m, runs=5)
    
    # Save performance statistics to file
    stats_file = f"{OUTPUT_DIR}/profile_stats_{dataset}_n{n}_m{m}.txt"
    with open(stats_file, 'w') as f:
        # Redirect to file
        stats.sort_stats('cumulative').stream = f
        stats.print_stats(20)  # Output top 20 functions
    
    print(f"  Single run time: {elapsed_time:.4f} seconds")
    print(f"  Detailed performance statistics saved to {stats_file}")
    
    return stats, elapsed_time

def test_system_load_impact(dataset="DS-1", n=DEFAULT_N, m=DEFAULT_M, b=100):
    """Test the impact of system load on performance"""
    print(f"\nTesting the impact of system load on performance (dataset={dataset}, n={n}, m={m}, b={b})")
    
    dataset_file = get_dataset_filename(dataset, n)
    grid, actual_n = initialize_grid(dataset_file)
    
    # Light load test
    print("  Running under light load...")
    light_start_time = time.time()
    light_cpu_start = time.process_time()
    
    _ = [hopkins_stat(grid, m) for _ in range(b)]
    
    light_cpu_time = time.process_time() - light_cpu_start
    light_wall_time = time.time() - light_start_time
    
    # Heavy load test (simulated by creating additional computation)
    print("  Running under heavy load...")
    load_threads = []
    for _ in range(3):
        # Create a background load process
        array_size = 5000
        matrices = [np.random.random((array_size, array_size)) for _ in range(3)]
        
    heavy_start_time = time.time()
    heavy_cpu_start = time.process_time()
    
    _ = [hopkins_stat(grid, m) for _ in range(b)]
    
    heavy_cpu_time = time.process_time() - heavy_cpu_start
    heavy_wall_time = time.time() - heavy_start_time
    
    # Save results
    results = {
        'dataset': dataset,
        'n': actual_n,
        'm': m,
        'B': b,
        'light_cpu_time': light_cpu_time,
        'light_wall_time': light_wall_time,
        'heavy_cpu_time': heavy_cpu_time,
        'heavy_wall_time': heavy_wall_time
    }
    
    print(f"  Light load: CPU time={light_cpu_time:.4f}s, Wall time={light_wall_time:.4f}s")
    print(f"  Heavy load: CPU time={heavy_cpu_time:.4f}s, Wall time={heavy_wall_time:.4f}s")
    print(f"  CPU time ratio (heavy/light): {heavy_cpu_time/light_cpu_time:.2f}")
    print(f"  Wall time ratio (heavy/light): {heavy_wall_time/light_wall_time:.2f}")
    
    # Save to file
    result_file = f"{OUTPUT_DIR}/system_load_{dataset}_n{n}_m{m}_b{b}.csv"
    with open(result_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    return results

def plot_results():
    """Draw experiment result charts"""
    plt.figure(figsize=(15, 10))
    
    # Draw B change chart
    try:
        for dataset in DATASETS:
            file_path = f"{OUTPUT_DIR}/varying_b_{dataset}_n{DEFAULT_N}_m{DEFAULT_M}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                plt.subplot(3, 1, 1)
                plt.plot(df['B'], df['time'], marker='o', label=f"{dataset}")
                plt.xlabel('Bootstrap number (B)')
                plt.ylabel('Running time (seconds)')
                plt.title('B value impact on running time')
                plt.legend()
                plt.grid(True)
    except Exception as e:
        print(f"Error drawing B change chart: {e}")
    
    # Draw m change chart
    try:
        for dataset in DATASETS:
            file_path = f"{OUTPUT_DIR}/varying_m_{dataset}_n{DEFAULT_N}_b{DEFAULT_B}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                plt.subplot(3, 1, 2)
                plt.plot(df['m'], df['time'], marker='o', label=f"{dataset}")
                plt.xlabel('Random point number (m)')
                plt.ylabel('Running time (seconds)')
                plt.title('m value impact on running time')
                plt.legend()
                plt.grid(True)
    except Exception as e:
        print(f"Error drawing m change chart: {e}")
    
    # Draw n change chart
    try:
        for dataset in DATASETS:
            file_path = f"{OUTPUT_DIR}/varying_n_{dataset}_m{DEFAULT_M}_b{DEFAULT_B}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                plt.subplot(3, 1, 3)
                plt.plot(df['n'], df['time'], marker='o', label=f"{dataset}")
                plt.xlabel('Dataset size (n)')
                plt.ylabel('Running time (seconds)')
                plt.title('n value impact on running time')
                plt.legend()
                plt.grid(True)
    except Exception as e:
        print(f"Error drawing n change chart: {e}")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/performance_results.png")
    plt.show()
    
    # Draw system load impact chart
    try:
        data = []
        for dataset in DATASETS:
            file_path = f"{OUTPUT_DIR}/system_load_{dataset}_n{DEFAULT_N}_m{DEFAULT_M}_b100.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                data.append(df.iloc[0])
        
        if data:
            df = pd.DataFrame(data)
            
            plt.figure(figsize=(12, 6))
            
            # CPU time comparison
            plt.subplot(1, 2, 1)
            bar_width = 0.35
            x = np.arange(len(df['dataset']))
            plt.bar(x - bar_width/2, df['light_cpu_time'], bar_width, label='Light load')
            plt.bar(x + bar_width/2, df['heavy_cpu_time'], bar_width, label='Heavy load')
            plt.xlabel('Dataset')
            plt.ylabel('CPU time (seconds)')
            plt.title('System load impact on CPU time')
            plt.xticks(x, df['dataset'])
            plt.legend()
            plt.grid(True)
            
            # Wall time comparison
            plt.subplot(1, 2, 2)
            plt.bar(x - bar_width/2, df['light_wall_time'], bar_width, label='Light load')
            plt.bar(x + bar_width/2, df['heavy_wall_time'], bar_width, label='Heavy load')
            plt.xlabel('Dataset')
            plt.ylabel('Wall time (seconds)')
            plt.title('System load impact on wall time')
            plt.xticks(x, df['dataset'])
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/system_load_impact.png")
            plt.show()
    except Exception as e:
        print(f"Error drawing system load impact chart: {e}")

def linear_regression_analysis():
    """Linear regression analysis of experimental data"""
    results_df = pd.DataFrame()
    
    # Collect all results
    for dataset in DATASETS:
        # B change data
        file_path = f"{OUTPUT_DIR}/varying_b_{dataset}_n{DEFAULT_N}_m{DEFAULT_M}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['varying'] = 'B'
            results_df = pd.concat([results_df, df])
        
        # m change data
        file_path = f"{OUTPUT_DIR}/varying_m_{dataset}_n{DEFAULT_N}_b{DEFAULT_B}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['varying'] = 'm'
            results_df = pd.concat([results_df, df])
        
        # n change data
        file_path = f"{OUTPUT_DIR}/varying_n_{dataset}_m{DEFAULT_M}_b{DEFAULT_B}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['varying'] = 'n'
            results_df = pd.concat([results_df, df])
    
    # If no data, return
    if results_df.empty:
        print("No data available for linear regression analysis")
        return
    
    # Linear regression by dataset and varying parameter
    regression_results = []
    
    for dataset in DATASETS:
        for varying in ['B', 'm', 'n']:
            df_subset = results_df[(results_df['dataset'] == dataset) & (results_df['varying'] == varying)]
            
            if not df_subset.empty:
                if varying == 'B':
                    X = df_subset['B'].values.reshape(-1, 1)
                    x_label = 'B'
                elif varying == 'm':
                    X = df_subset['m'].values.reshape(-1, 1)
                    x_label = 'm'
                else:  # varying == 'n'
                    X = df_subset['n'].values.reshape(-1, 1)
                    x_label = 'n'
                
                y = df_subset['time'].values
                
                # Linear regression
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate R²
                from sklearn.metrics import r2_score
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                # Calculate 95% confidence interval
                import scipy.stats as stats
                from sklearn.metrics import mean_squared_error
                n = len(X)
                p = 1  # Feature count
                dof = n - p - 1  # Degrees of freedom
                mse = mean_squared_error(y, y_pred)
                
                t_critical = stats.t.ppf(0.975, dof)  # t value for 95% confidence interval
                std_error = np.sqrt(mse / np.sum((X - np.mean(X))**2))
                
                ci_lower = model.coef_[0] - t_critical * std_error
                ci_upper = model.coef_[0] + t_critical * std_error
                
                regression_results.append({
                    'dataset': dataset,
                    'varying': varying,
                    'coefficient': model.coef_[0],
                    'intercept': model.intercept_,
                    'r2': r2,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
                
                # Draw regression line and data points
                plt.figure(figsize=(8, 6))
                plt.scatter(X, y, color='blue', alpha=0.7)
                plt.plot(X, y_pred, color='red', linewidth=2)
                
                plt.fill_between(
                    X.flatten(),
                    y_pred - t_critical * std_error * np.sqrt(1 + 1/n + (X.flatten() - np.mean(X.flatten()))**2 / np.sum((X.flatten() - np.mean(X.flatten()))**2)),
                    y_pred + t_critical * std_error * np.sqrt(1 + 1/n + (X.flatten() - np.mean(X.flatten()))**2 / np.sum((X.flatten() - np.mean(X.flatten()))**2)),
                    color='gray', alpha=0.2
                )
                
                plt.xlabel(f'{x_label}')
                plt.ylabel('Time (seconds)')
                plt.title(f'Dataset {dataset} - Linear regression ({x_label} change)')
                plt.grid(True)
                plt.savefig(f"{OUTPUT_DIR}/regression_{dataset}_{varying}.png")
                plt.close()
    
    # Save regression results
    regression_df = pd.DataFrame(regression_results)
    regression_df.to_csv(f"{OUTPUT_DIR}/regression_results.csv", index=False)
    
    return regression_df

def main():
    """Main function, run all tests"""
    print("Starting performance test...")
    
    # Run tests for each dataset
    for dataset in DATASETS:
        # 1. Detailed performance analysis (single function call)
        test_single_run_profiling(dataset=dataset)
        
        # 2. Test varying B
        test_varying_b(dataset=dataset)
        
        # 3. Test varying m
        test_varying_m(dataset=dataset)
        
        # 4. Test varying n (use smaller B value to speed up test)
        test_varying_n(dataset=dataset, b=100)
        
        # 5. System load test (use smaller B value to speed up test)
        test_system_load_impact(dataset=dataset)
    
    # Draw result charts
    plot_results()
    
    # Linear regression analysis
    linear_regression_analysis()
    
    print("\nAll tests completed! Results saved to performance_results directory.")

if __name__ == "__main__":
    import sys
    
    # Output system information
    import platform
    print("System information:")
    print(f"   Operating system: {platform.system()} {platform.version()}")
    print(f"   Python version: {platform.python_version()}")
    print(f"   Processor: {platform.processor()}")
    
    # If you want to run specific tests, you can specify them via command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "profile":
            for dataset in DATASETS:
                test_single_run_profiling(dataset=dataset)
        elif sys.argv[1] == "b":
            for dataset in DATASETS:
                test_varying_b(dataset=dataset)
        elif sys.argv[1] == "m":
            for dataset in DATASETS:
                test_varying_m(dataset=dataset)
        elif sys.argv[1] == "n":
            for dataset in DATASETS:
                test_varying_n(dataset=dataset, b=100)
        elif sys.argv[1] == "load":
            for dataset in DATASETS:
                test_system_load_impact(dataset=dataset)
        elif sys.argv[1] == "plot":
            plot_results()
        elif sys.argv[1] == "regression":
            linear_regression_analysis()
    else:
        # Run all tests
        main() 