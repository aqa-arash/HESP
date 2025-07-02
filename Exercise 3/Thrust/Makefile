# Compiler and flags
NVCC := nvcc
NVCCFLAGS := -O3 
RM := rm -f

# Get architecture
ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | sed 's/\.//')
NVARCH := sm_$(ARCH)
NVCCFLAGS += -gencode arch=compute_$(ARCH),code=$(NVARCH)
NVCCFLAGS += -lcublas --extended-lambda -std=c++17 --compiler-bindir=/usr/bin/gcc-9 -lcublas -lm -lstdc++

# Target executables
TARGET := md
INIT_TARGET := initialVTKmaker

# Source files
SRCS := md.cu
INIT_SRCS := initialVTKmaker.cpp

# Header files
HEADERS := parser.hpp cudafuncs.hpp cpufuncs.hpp 

# Object files
OBJS := $(SRCS:.cu=.o)
INIT_OBJS := $(INIT_SRCS:.cpp=.o)

# Build targets
all: $(TARGET) $(INIT_TARGET)

# Build md executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Build initialVTKmaker executable
$(INIT_TARGET): $(INIT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile CUDA source files to object files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	$(RM) $(OBJS) $(INIT_OBJS) $(TARGET) $(INIT_TARGET)

.PHONY: clean all