# Compiler and flags
COMPILER := g++
COMPILERFLAGS := -O3 -std=c++20 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart 
RM := rm -f
#add flags requried for std par and cuda 
# Get architecture
#ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | sed 's/\.//')
#NVARCH := sm_$(ARCH)
#NVCCFLAGS += -gencode arch=compute_$(ARCH),code=$(NVARCH)
#NVCCFLAGS += -lcublas

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

# Build md executable
$(TARGET): $(OBJS)
	$(COMPILER) $(COMPILERFLAGS) -o $@ $^

# Build initialVTKmaker executable
$(INIT_TARGET): $(INIT_OBJS)
	$(COMPILER) $(COMPILERFLAGS) -o $@ $^

# Compile C++ source files to object files (default)
%.o: %.cpp $(HEADERS)
	$(COMPILER) $(COMPILERFLAGS) -c $< -o $@

