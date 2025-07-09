# Compiler and flags - Use NVIDIA HPC SDK nvc++ for A100
COMPILER := nvc++

# A100 specific flags (compute capability 8.0)
# -mp=gpu für OpenMP GPU offloading
# -gpu=cc80 für A100 architecture
# -Minfo=mp für detaillierte OpenMP Informationen
COMPULERCFLAGS := -O3 -mp=gpu -gpu=cc80 -fPIC -std=c++17

# Alternative falls du CUDA direkt verwenden möchtest:
# COMPULERCFLAGS := -O3 -mp=gpu -gpu=cc80,cuda11.8 -Minfo=mp -fPIC

# GPU architecture für A100 (compute capability 8.0)
GPU_ARCH := sm_80

# Linker flags - OpenMP GPU support
LDFLAGS := -mp=gpu -gpu=cc80

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
	$(COMPILER) $(COMPULERCFLAGS) $(LDFLAGS) -o $@ $^ -lstdc++fs

# Build initialVTKmaker executable
$(INIT_TARGET): $(INIT_OBJS)
	$(COMPILER) $(COMPULERCFLAGS) -o $@ $^ -lstdc++fs

# Compile C++ source files to object files with OpenMP GPU support
%.o: %.cpp $(HEADERS)
	$(COMPILER) $(COMPULERCFLAGS) -c $< -o $@

# Clean up build files
clean:
	$(RM) $(OBJS) $(INIT_OBJS) $(TARGET) $(INIT_TARGET)

# Debug target - zeigt detaillierte Compiler-Informationen
debug: COMPULERCFLAGS += -g -Minfo=all
debug: $(TARGET)

# Profiling target - für Performance-Analyse
profile: COMPULERCFLAGS += -pg
profile: LDFLAGS += -pg
profile: $(TARGET)

.PHONY: all clean debug profile