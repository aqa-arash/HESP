# Compiler and flags
COMPILER := icpx
COMPULERCFLAGS :=  -std=c++17
RM := rm -f

# Linker flags
LDFLAGS := -std=c++17 -lstdc++fs

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
	$(COMPILER) $(COMPULERCFLAGS) $(LDFLAGS) -o $@ $^ -lstdc++fs

# Build initialVTKmaker executable
$(INIT_TARGET): $(INIT_OBJS)
	$(COMPILER) $(COMPULERCFLAGS) $(LDFLAGS) -o $@ $^ -lstdc++fs

# Compile CUDA source files to object files
%.o: %.cpp $(HEADERS)
	$(COMPILER) $(COMPULERCFLAGS) -c $< -o $@

# Compile C++ source files to object files
%.o: %.cpp $(HEADERS)
	$(COMPILER) $(COMPULERCFLAGS) -c $< -o $@

# Clean up build files
clean:
	$(RM) $(OBJS) $(INIT_OBJS) $(TARGET) $(INIT_TARGET)

.PHONY: clean all