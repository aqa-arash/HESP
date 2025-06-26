# Compiler and flags
COMPILER := g++
COMPULERCFLAGS :=  
RM := rm -f

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
	$(COMPILER) $(COMPULERCFLAGS) -o $@ $^

# Build initialVTKmaker executable
$(INIT_TARGET): $(INIT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile CUDA source files to object files
%.o: %.cpp $(HEADERS)
	$(COMPILER) $(COMPULERCFLAGS) -c $< -o $@

# Compile C++ source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	$(RM) $(OBJS) $(INIT_OBJS) $(TARGET) $(INIT_TARGET)

.PHONY: clean all