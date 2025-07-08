#!/usr/bin/env python3
"""
Performance Analysis Plotter for MD Simulation Comparison
Compares CUDA, OpenMPgpu, and CPU-only versions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def get_next_filename(output_dir, base_name, extension='.png'):
    """
    Get the next available numbered filename in the output directory
    
    Args:
        output_dir (Path): Output directory path
        base_name (str): Base name for the file (without extension)
        extension (str): File extension (default: '.png')
        
    Returns:
        Path: Next available numbered filename
    """
    counter = 1
    while True:
        filename = f"{base_name}_{counter:03d}{extension}"
        filepath = output_dir / filename
        if not filepath.exists():
            return filepath
        counter += 1

def load_performance_data(file_path):
    """
    Load performance data from CSV file
    
    Args:
        file_path (str): Path to the performance results file
        
    Returns:
        pandas.DataFrame or None: Loaded data or None if file doesn't exist
    """
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} data points from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    else:
        print(f"File not found: {file_path}")
        return None

def create_performance_plot():
    """
    Create performance comparison plot with logarithmic axes for total time
    """
    # Create output directory if it doesn't exist
    output_dir = Path('performance_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Define file paths
    files = {
        'CUDA': 'cuda/cuda_performance_results.txt',
        'OpenMPgpu': 'omp/ompGPU_performance_results.txt',
        'CPU Only': 'decudafied/decudafied_performance_results.txt'
    }
    
    # Load data from all files
    data = {}
    for label, file_path in files.items():
        df = load_performance_data(file_path)
        if df is not None:
            data[label] = df
    
    if not data:
        print("No data files could be loaded. Please check file paths.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Colors for different versions
    colors = {'CUDA': '#1f77b4', 'OpenMPgpu': '#ff7f0e', 'CPU Only': '#2ca02c'}
    markers = {'CUDA': 'o', 'OpenMPgpu': 's', 'CPU Only': '^'}
    
    # Plot data for each version
    for label, df in data.items():
        # Use Total_Time_s if available (CUDA), otherwise use Total_Time_s
        if 'Total_GPU_Time_s' in df.columns:
            y_data = df['Total_GPU_Time_s']
            time_label = 'Total GPU Time'
        else:
            y_data = df['Total_Time_s']
            time_label = 'Total Time'
            
        plt.loglog(df['Particles'], y_data, 
                  marker=markers[label], 
                  color=colors[label], 
                  linewidth=2, 
                  markersize=8,
                  label=f'{label}',
                  alpha=0.8)
        
        print(f"{label}: {len(df)} particles, max particles: {df['Particles'].max()}")
    
    # Customize the plot
    plt.xlabel('Number of Particles', fontsize=14, fontweight='bold')
    plt.ylabel('Running Time [s]', fontsize=14, fontweight='bold')
    plt.title('MD Simulation Performance Comparison - Total Time\n(Log-Log Scale)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    
    # Legend
    plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
    
    # Set axis limits and ticks
    if data:
        all_particles = []
        all_times = []
        
        for df in data.values():
            all_particles.extend(df['Particles'].tolist())
            if 'Total_GPU_Time_s' in df.columns:
                all_times.extend(df['Total_GPU_Time_s'].tolist())
            else:
                all_times.extend(df['Total_Time_s'].tolist())
        
        plt.xlim(min(all_particles) * 0.8, max(all_particles) * 1.2)
        plt.ylim(min(all_times) * 0.8, max(all_times) * 1.2)
    
    # Add reference lines for scaling analysis
    if data:
        x_ref = np.array([10, 50000])
        
        # O(N) reference
        y_on = x_ref * (1.0 / 10)  # Normalized to pass through a reasonable point
        plt.loglog(x_ref, y_on, '--', color='gray', alpha=0.6, linewidth=1, label='O(N)')
        
        # O(N²) reference
        y_on2 = (x_ref ** 2) * (1.0 / 100)  # Normalized
        plt.loglog(x_ref, y_on2, '--', color='red', alpha=0.6, linewidth=1, label='O(N²)')
    
    plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot with auto-numbering
    output_file = get_next_filename(output_dir, 'total_time_comparison')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nTotal time plot saved as: {output_file}")
    
    # Show plot
    plt.show()

def create_detailed_performance_plots():
    """
    Create detailed performance comparison plots for individual components
    """
    # Create output directory if it doesn't exist
    output_dir = Path('performance_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Define file paths
    files = {
        'CUDA': 'cuda/cuda_performance_results.txt',
        'OpenMPgpu': 'omp/ompGPU_performance_results.txt',
        'CPU Only': 'decudafied/decudafied_performance_results.txt'
    }
    
    # Load data from all files
    data = {}
    for label, file_path in files.items():
        df = load_performance_data(file_path)
        if df is not None:
            data[label] = df
    
    if not data:
        print("No data files could be loaded for detailed plots.")
        return
    
    # Define the components to plot
    components = {
        'Update_Pos_s': 'Position Update Time',
        'Update_Vel_s': 'Velocity Update Time', 
        'Update_Forces_Acc_s': 'Forces & Acceleration Time'
    }
    
    # Colors and markers
    colors = {'CUDA': '#1f77b4', 'OpenMPgpu': '#ff7f0e', 'CPU Only': '#2ca02c'}
    markers = {'CUDA': 'o', 'OpenMPgpu': 's', 'CPU Only': '^'}
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('MD Simulation Performance Comparison - Component Analysis\n(Log-Log Scale)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Plot each component
    for idx, (component, title) in enumerate(components.items()):
        ax = axes[idx]
        
        # Plot data for each version
        for label, df in data.items():
            if component in df.columns:
                ax.loglog(df['Particles'], df[component], 
                         marker=markers[label], 
                         color=colors[label], 
                         linewidth=2, 
                         markersize=6,
                         label=f'{label}',
                         alpha=0.8)
        
        # Customize subplot
        ax.set_xlabel('Number of Particles', fontsize=12, fontweight='bold')
        ax.set_ylabel('Running Time [s]', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Grid
        ax.grid(True, which="both", ls="-", alpha=0.3)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
        
        # Legend (only on first subplot to avoid clutter)
        if idx == 0:
            ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
        
        # Add reference lines for scaling analysis
        if data:
            all_particles = []
            all_times = []
            
            for df in data.values():
                if component in df.columns:
                    all_particles.extend(df['Particles'].tolist())
                    all_times.extend(df[component].tolist())
            
            if all_particles and all_times:
                x_ref = np.array([min(all_particles), max(all_particles)])
                
                # O(N) reference
                y_on = x_ref * (min(all_times) / min(all_particles))
                ax.loglog(x_ref, y_on, '--', color='gray', alpha=0.5, linewidth=1)
                
                # O(N²) reference  
                y_on2 = (x_ref ** 2) * (min(all_times) / (min(all_particles) ** 2))
                ax.loglog(x_ref, y_on2, '--', color='red', alpha=0.5, linewidth=1)
                
                # Add labels only on the last subplot
                if idx == 2:
                    ax.loglog([], [], '--', color='gray', alpha=0.5, linewidth=1, label='O(N)')
                    ax.loglog([], [], '--', color='red', alpha=0.5, linewidth=1, label='O(N²)')
                    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot with auto-numbering
    output_file = get_next_filename(output_dir, 'component_analysis_comparison')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Component analysis plot saved as: {output_file}")
    
    # Show plot
    plt.show()

def print_performance_summary():
    """
    Print a summary of performance data
    """
    files = {
        'CUDA': 'cuda/cuda_performance_results.txt',
        'OpenMPgpu': 'omp/ompGPU_performance_results.txt',
        'CPU Only': 'decudafied/decudafied_performance_results.txt'
    }
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for label, file_path in files.items():
        df = load_performance_data(file_path)
        if df is not None:
            if 'Total_GPU_Time_s' in df.columns:
                time_col = 'Total_GPU_Time_s'
            else:
                time_col = 'Total_Time_s'
                
            min_time = df[time_col].min()
            max_time = df[time_col].max()
            min_particles = df['Particles'].min()
            max_particles = df['Particles'].max()
            
            print(f"\n{label}:")
            print(f"  Particle range: {min_particles} - {max_particles}")
            print(f"  Time range: {min_time:.3f}s - {max_time:.3f}s")
            print(f"  Data points: {len(df)}")
            
            # Performance at common particle counts
            common_particles = [1000, 10000]
            for n in common_particles:
                row = df[df['Particles'] == n]
                if not row.empty:
                    time_val = row[time_col].iloc[0]
                    print(f"  Time at {n} particles: {time_val:.3f}s")

if __name__ == "__main__":
    print("MD Simulation Performance Analysis")
    print("=" * 40)
    
    # Print summary
    print_performance_summary()
    
    # Create total time comparison plot
    print("\nCreating total time comparison plot...")
    create_performance_plot()
    
    # Create detailed component analysis plots
    print("\nCreating detailed component analysis plots...")
    create_detailed_performance_plots()
    
    print("\nAnalysis complete!")
    print("All plots have been saved to the 'performance_plots' directory.")