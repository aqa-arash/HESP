# Compiler and flags
# Intel icpx compiler (C++ specific Intel compiler - recommended)
COMPILER := icpx
# Simple OpenMP flags (let compiler auto-detect GPU)
COMPULERCFLAGS := -O3 -qopenmp -fPIC
# OpenMP flags for GPU offloading with Intel compiler (NVIDIA GPU)
# COMPULERCFLAGS := -O3 -qopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fPIC
# Alternative for Intel GPUs:
# COMPULERCFLAGS := -O3 -qopenmp -fopenmp-targets=spir64 -fPIC

# Fallback to GCC if Intel compiler not available
# COMPILER := g++
# COMPULERCFLAGS := -O3 -fopenmp -foffload=nvptx-none -fPIC

# CUDA/GPU specific settings for RTX 2060 mobile (Turing architecture, compute capability 7.5)
GPU_ARCH := sm_75

# Linker flags
LDFLAGS := -qopenmp -fPIE

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
	@$(COMPILER) -fopenmp -foffload=nvptx-none --version
	@echo "GPU Architecture: $(GPU_ARCH)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build both executables"
	@echo "  $(TARGET)    - Build md executable with OpenMP GPU support"
	@echo "  $(INIT_TARGET) - Build initialVTKmaker executable"
	@echo "  clean        - Remove all build files"
	@echo "  check-openmp - Check OpenMP GPU support"
	@echo "  help         - Show this help message"

.PHONY: clean all check-openmp help