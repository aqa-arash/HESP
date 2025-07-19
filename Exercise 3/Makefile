# Compiler - NVIDIA HPC SDK
COMPILER := nvc++

# GPU architecture for A100 = compute capability 8.0 (sm_80)
GPU_ARCH := cc80

# OpenACC compiler flags
COMPILERFLAGS := -O3 -acc -gpu=$(GPU_ARCH) -fPIC -std=c++17 -Minfo=accel

# Linker flags (same as compiler flags for simplicity)
LDFLAGS := $(COMPILERFLAGS)

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

# Default rule: build both targets
all: $(TARGET) $(INIT_TARGET)

# Build md executable
$(TARGET): $(OBJS)
	$(COMPILER) $(LDFLAGS) -o $@ $^ -lstdc++fs

# Build initialVTKmaker executable
$(INIT_TARGET): $(INIT_OBJS)
	$(COMPILER) $(LDFLAGS) -o $@ $^ -lstdc++fs

# Compile .cpp to .o
%.o: %.cpp $(HEADERS)
	$(COMPILER) $(COMPILERFLAGS) -c $< -o $@

# Clean up build files
clean:
	$(RM) $(OBJS) $(INIT_OBJS) $(TARGET) $(INIT_TARGET)

# Debug target - shows detailed compiler info
debug: COMPILERFLAGS += -g -Minfo=all
debug: LDFLAGS += -g
debug: clean all

# Profiling target - for performance analysis
profile: COMPILERFLAGS += -pg
profile: LDFLAGS += -pg
profile: clean all

.PHONY: all clean debug profile
