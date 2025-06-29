#!/bin/bash

# Performance Analysis Script for OpenMP GPU Code

# Define particle counts for testing
particle_counts=(10 20 50 100 200 500 1000 2000 5000 10000 20000)

# Output file for results
output_file="decudafied_performance_results.txt"

# Create header for results file
echo "Particles,Update_Pos_s,Update_Vel_s,Update_Forces_Acc_s,Total_Time_s" > "$output_file"

echo "=== Performance Analysis started ==="
echo "Results will be saved to $output_file"
echo

# Loop through all particle counts
for num_particles in "${particle_counts[@]}"; do
    echo "Testing with $num_particles particles..."
    
    # 1. Create VTK file
    echo "  - Creating VTK file for $num_particles particles"
    ./initialVTKmaker "$num_particles"
    
    if [ $? -ne 0 ]; then
        echo "  ERROR: Could not create VTK file for $num_particles particles"
        continue
    fi
    
    # 2. Update config.txt
    echo "  - Updating config.txt"
    cat > config.txt << EOF
# Configuration file for simulation
# Lines starting with '#' are comments and will be ignored

initialization file: initial/initial${num_particles}_500.vtk

box size: 500
time step length: 0.05
time step count: 50000
sigma: 3.4
epsilon: 0.238

print interval: 0
cutoff radius: 8.0
acceleration: 1
EOF
    
    # 3. Run simulation and extract times
    echo "  - Running simulation..."
    output=$(./md config.txt 2>&1)
    
    if [ $? -ne 0 ]; then
        echo "  ERROR: Simulation failed for $num_particles particles"
        echo "$output"
        continue
    fi
    
    # 4. Extract time values from output
    update_pos_time=$(echo "$output" | grep "Time for.*updating positions" | sed -n 's/.*: \([0-9.]*\) seconds/\1/p')
    update_vel_time=$(echo "$output" | grep "Time for.*updating velocities" | sed -n 's/.*: \([0-9.]*\) seconds/\1/p')
    forces_acc_time=$(echo "$output" | grep "Time for.*forces and accelerations" | sed -n 's/.*: \([0-9.]*\) seconds/\1/p')
    total_time=$(echo "$output" | grep "Total time for simulation" | sed -n 's/.*: \([0-9.]*\) seconds/\1/p')
    
    # 5. Check if all values were extracted
    if [[ -z "$update_pos_time" || -z "$update_vel_time" || -z "$forces_acc_time" || -z "$total_time" ]]; then
        echo "  ERROR: Could not extract all time values for $num_particles particles"
        echo "  Debug output:"
        echo "$output"
        continue
    fi
    
    # 6. Write results to file
    echo "$num_particles,$update_pos_time,$update_vel_time,$forces_acc_time,$total_time" >> "$output_file"
    
    echo "  - Success: Pos=${update_pos_time}s, Vel=${update_vel_time}s, Forces=${forces_acc_time}s, Total=${total_time}s"
    echo
done

echo "=== Performance Analysis completed ==="
echo "Results saved to: $output_file"
echo
echo "First 5 lines of results:"
head -n 6 "$output_file"

# Optional: Show brief statistics
echo
echo "=== Brief Statistics ==="
if [ -f "$output_file" ]; then
    total_tests=$(tail -n +2 "$output_file" | wc -l)
    echo "Number of successful tests: $total_tests"
    
    if [ $total_tests -gt 0 ]; then
        echo "Fastest Total Time: $(tail -n +2 "$output_file" | cut -d',' -f5 | sort -n | head -n1)s"
        echo "Slowest Total Time: $(tail -n +2 "$output_file" | cut -d',' -f5 | sort -n | tail -n1)s"
    fi
fi

echo
echo "The file $output_file can now be used for plotting."