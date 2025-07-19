# Compiler and flags - Use GCC for NVIDIA GPU
COMPILER := g++
# WICHTIG: -fno-stack-protector hinzugefügt um __stack_chk_guard Fehler zu vermeiden
COMPULERCFLAGS := -O3 -fopenmp -foffload=nvptx-none -foffload="-lm -fno-stack-protector" -fPIC -fcf-protection=none -fno-stack-protector

# Intel icpx fallback (commented out due to NVPTX issues)
# COMPILER := icpx
# COMPULERCFLAGS := -O3 -qopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fPIC

# CUDA/GPU specific settings for RTX 2060 mobile (Turing architecture, compute capability 7.5)
GPU_ARCH := sm_75

# Linker flags - wichtig: OpenMP muss auch beim Linken dabei sein
LDFLAGS := -fopenmp -fPIE

# Remove tool
RM := rm -f

# Target executables
TARGET := md
INIT_TARGET := initialVTKmaker

# Source files
SRCS := md.cpp
INIT_SRCS := initialVTKmaker.cpp

# Header files
HEADERS := parser.hpp cudafuncs.hpp cpufuncs.hpp 

# Object files
OBJS := $(SRCS:.cpp=.o)
INIT_OBJS := $(INIT_SRCS:.cpp=.o)

# Build targets
all: $(TARGET) $(INIT_TARGET)

# Build md executable with OpenMP GPU support
$(TARGET): $(OBJS)
	$(COMPILER) $(COMPULERCFLAGS) $(LDFLAGS) -o $@ $^

# Build initialVTKmaker executable
$(INIT_TARGET): $(INIT_OBJS)
	$(COMPILER) $(COMPULERCFLAGS) -o $@ $^

# Compile C++ source files to object files with OpenMP GPU support
%.o: %.cpp $(HEADERS)
	$(COMPILER) $(COMPULERCFLAGS) -c $< -o $@

# Clean up build files
clean:
	$(RM) $(OBJS) $(INIT_OBJS) $(TARGET) $(INIT_TARGET)

# Debug target to check OpenMP GPU support
check-openmp:
	@echo "Checking OpenMP GPU support..."
	@echo "Compiler: $(COMPILER)"
	@$(COMPILER) --version
	@echo ""
	@echo "Checking offload support:"
	@$(COMPILER) -v 2>&1 | grep -i offload || echo "No offload support found"
	@echo ""
	@echo "Checking NVPTX tools:"
	@which nvptx-none-as 2>/dev/null || echo "nvptx-none-as not found"
	@which nvptx-none-ld 2>/dev/null || echo "nvptx-none-ld not found"
	@echo ""
	@echo "GPU Architecture: $(GPU_ARCH)"
	@echo "Checking NVIDIA driver..."
	@nvidia-smi 2>/dev/null || echo "nvidia-smi not found - NVIDIA drivers may not be installed"

# Test OpenMP target offloading - ERWEITERT mit Stack-Protection Test
test-gpu:
	@echo "Testing OpenMP GPU offloading with stack protection disabled..."
	@echo '#include <omp.h>' > test_omp.cpp
	@echo '#include <iostream>' >> test_omp.cpp
	@echo '#include <vector>' >> test_omp.cpp
	@echo 'int main() {' >> test_omp.cpp
	@echo '  int num_devices = omp_get_num_devices();' >> test_omp.cpp
	@echo '  printf("Number of devices: %d\\n", num_devices);' >> test_omp.cpp
	@echo '  const int N = 1000;' >> test_omp.cpp
	@echo '  double a[N], b[N], c[N];' >> test_omp.cpp
	@echo '  for(int i=0; i<N; i++) { a[i] = i; b[i] = i*2; }' >> test_omp.cpp
	@echo '  #pragma omp target teams distribute parallel for map(to:a[0:N],b[0:N]) map(from:c[0:N])' >> test_omp.cpp
	@echo '  for(int i=0; i<N; i++) {' >> test_omp.cpp
	@echo '    c[i] = a[i] + b[i];' >> test_omp.cpp
	@echo '  }' >> test_omp.cpp
	@echo '  printf("Test result: c[500] = %f (should be 1500)\\n", c[500]);' >> test_omp.cpp
	@echo '  return 0;' >> test_omp.cpp
	@echo '}' >> test_omp.cpp
	@$(COMPILER) $(COMPULERCFLAGS) $(LDFLAGS) test_omp.cpp -o test_omp
	@./test_omp
	@$(RM) test_omp.cpp test_omp

# NEUES TARGET: Test mit std::vector (dein ursprüngliches Problem)
test-vector-gpu:
	@echo "Testing std::vector with GPU offloading..."
	@echo '#include <omp.h>' > test_vector.cpp
	@echo '#include <iostream>' >> test_vector.cpp
	@echo '#include <vector>' >> test_vector.cpp
	@echo 'void update_velocities_test(std::vector<double>& vel_new, const std::vector<double>& vel_old, const std::vector<double>& acc, double dt, int numParticles) {' >> test_vector.cpp
	@echo '  double* v_new = vel_new.data();' >> test_vector.cpp
	@echo '  double* v_old = const_cast<double*>(vel_old.data());' >> test_vector.cpp
	@echo '  double* a = const_cast<double*>(acc.data());' >> test_vector.cpp
	@echo '  int total_size = numParticles * 3;' >> test_vector.cpp
	@echo '  #pragma omp target teams distribute parallel for map(to:v_old[0:total_size],a[0:total_size]) map(tofrom:v_new[0:total_size])' >> test_vector.cpp
	@echo '  for(int i = 0; i < total_size; ++i) {' >> test_vector.cpp
	@echo '    v_new[i] = v_old[i] + 0.5 * a[i] * dt;' >> test_vector.cpp
	@echo '  }' >> test_vector.cpp
	@echo '}' >> test_vector.cpp
	@echo 'int main() {' >> test_vector.cpp
	@echo '  const int numParticles = 1000;' >> test_vector.cpp
	@echo '  std::vector<double> vel_new(numParticles*3, 0.0);' >> test_vector.cpp
	@echo '  std::vector<double> vel_old(numParticles*3);' >> test_vector.cpp
	@echo '  std::vector<double> acc(numParticles*3);' >> test_vector.cpp
	@echo '  for(int i=0; i<numParticles*3; i++) { vel_old[i] = i*0.1; acc[i] = i*0.01; }' >> test_vector.cpp
	@echo '  update_velocities_test(vel_new, vel_old, acc, 0.01, numParticles);' >> test_vector.cpp
	@echo '  printf("GPU vector test result: vel_new[1500] = %f\\n", vel_new[1500]);' >> test_vector.cpp
	@echo '  return 0;' >> test_vector.cpp
	@echo '}' >> test_vector.cpp
	@$(COMPILER) $(COMPULERCFLAGS) $(LDFLAGS) test_vector.cpp -o test_vector
	@./test_vector
	@$(RM) test_vector.cpp test_vector

# Intel GPU target
intel-gpu:
	@echo "Building with Intel GPU support..."
	$(MAKE) COMPULERCFLAGS="-O3 -qopenmp -fopenmp-targets=spir64 -fPIC" LDFLAGS="-qopenmp -fPIE" all

# Test Intel GPU
test-intel-gpu:
	@echo "Testing Intel GPU offloading..."
	@echo '#include <omp.h>' > test_intel.cpp
	@echo '#include <iostream>' >> test_intel.cpp
	@echo 'int main() {' >> test_intel.cpp
	@echo '  int num_devices = omp_get_num_devices();' >> test_intel.cpp
	@echo '  printf("Number of devices: %d\\n", num_devices);' >> test_intel.cpp
	@echo '  #pragma omp target' >> test_intel.cpp
	@echo '  {' >> test_intel.cpp
	@echo '    if (omp_is_initial_device()) printf("Running on CPU\\n"); else printf("Running on Intel GPU\\n");' >> test_intel.cpp
	@echo '  }' >> test_intel.cpp
	@echo '  return 0;' >> test_intel.cpp
	@echo '}' >> test_intel.cpp
	@icpx -O3 -qopenmp -fopenmp-targets=spir64 -fPIC -qopenmp -fPIE test_intel.cpp -o test_intel
	@./test_intel
	@$(RM) test_intel.cpp test_intel

cpu-only:
	@echo "Building CPU-only version..."
	$(MAKE) COMPULERCFLAGS="-O3 -fopenmp -fPIC" all

# Test CPU OpenMP (without GPU offloading)
test-cpu:
	@echo "Testing CPU OpenMP..."
	@echo '#include <omp.h>' > test_cpu.cpp
	@echo '#include <iostream>' >> test_cpu.cpp
	@echo 'int main() {' >> test_cpu.cpp
	@echo '  #pragma omp parallel' >> test_cpu.cpp
	@echo '  {' >> test_cpu.cpp
	@echo '    printf("Thread %d of %d\\n", omp_get_thread_num(), omp_get_num_threads());' >> test_cpu.cpp
	@echo '  }' >> test_cpu.cpp
	@echo '  return 0;' >> test_cpu.cpp
	@echo '}' >> test_cpu.cpp
	@g++ -fopenmp test_cpu.cpp -o test_cpu
	@./test_cpu
	@$(RM) test_cpu.cpp test_cpu

help:
	@echo "Available targets:"
	@echo "  all          - Build both executables (NVIDIA GPU)"
	@echo "  $(TARGET)    - Build md executable with OpenMP GPU support"
	@echo "  $(INIT_TARGET) - Build initialVTKmaker executable"
	@echo "  cpu-only     - Build CPU-only version"
	@echo "  intel-gpu    - Build with Intel GPU support"
	@echo "  clean        - Remove all build files"
	@echo "  check-openmp - Check OpenMP GPU support"
	@echo "  test-gpu     - Test NVIDIA GPU offloading (basic arrays)"
	@echo "  test-vector-gpu - Test std::vector with GPU offloading"
	@echo "  test-intel-gpu - Test Intel GPU offloading"
	@echo "  test-cpu     - Test CPU OpenMP"
	@echo "  help         - Show this help message"

.PHONY: clean all check-openmp test-gpu test-vector-gpu test-cpu cpu-only intel-gpu test-intel-gpu help