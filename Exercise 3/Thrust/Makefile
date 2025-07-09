# Compiler and flags (use environment variables or defaults)
NVCC ?= nvcc
CXX ?= g++
NVCCFLAGS ?= -O3
CXXFLAGS ?= -O3
RM ?= rm -f

# CUDA toolkit path (optional, set externally if needed)

# CUDA compiler flags for nvcc
NVCCFLAGS += --extended-lambda -std=c++17 -lcublas -lm -lstdc++

# Alternative: If you want to use nvc++ instead of nvcc
# NVCC ?= nvc++
# NVCCFLAGS ?= -O3 -std=c++17 -cuda -gpu=cc80 -lcublas -lm -lstdc++

# Optionally set compiler-bindir if needed
ifdef GCC_BIN
    NVCCFLAGS += --compiler-bindir=$(GCC_BIN)
endif

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