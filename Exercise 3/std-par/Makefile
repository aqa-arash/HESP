
LIBS     = -L$(CUDA_HOME)/lib64 -lcudart
# Compiler
COMPILER := nvc++
FLAGS := -std=c++17 -O3 -stdpar -pthread -lstdc++fs  $(LIBS)

# Targets
TARGET := md-gpu
INIT_TARGET := initialVTKmaker

# Source files
SRCS := md.cpp
INIT_SRCS := initialVTKmaker.cpp
HEADERS := parser.hpp cudafuncs.hpp cpufuncs.hpp

# Object files
OBJS := $(SRCS:.cpp=.o)
INIT_OBJS := $(INIT_SRCS:.cpp=.o)

# Default target
all: $(TARGET) $(INIT_TARGET)

$(TARGET): $(OBJS)
	$(COMPILER) $(FLAGS) -o $@ $^

$(INIT_TARGET): $(INIT_OBJS)
	$(COMPILER) $(FLAGS) -o $@ $^

%.o: %.cpp $(HEADERS)
	$(COMPILER) $(FLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET) $(INIT_TARGET)