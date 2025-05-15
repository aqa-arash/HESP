# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -Wall -std=c++17
RM := rm -f

# Target executables
TARGET := md
INIT_TARGET := initialVTKmaker

# Source files
SRCS := md.cpp
INIT_SRCS := initialVTKmaker.cpp

# Header files
HEADERS := parser.hpp

# Object files
OBJS := $(SRCS:.cpp=.o)
INIT_OBJS := $(INIT_SRCS:.cpp=.o)

# Build targets
all: $(TARGET) $(INIT_TARGET)

# Build md executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build initialVTKmaker executable
$(INIT_TARGET): $(INIT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	$(RM) $(OBJS) $(INIT_OBJS) $(TARGET) $(INIT_TARGET)

.PHONY: clean all