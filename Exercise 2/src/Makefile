# Compiler and flags
CXX = g++
NVCC = nvcc

CXXFLAGS = -lm
LODEPNG = ./lodepng/lodepng.cpp

# Targets
all: julia_png julia_png_cuda

julia_png: julia_png.c $(LODEPNG)
	$(CXX) -o $@ $^ $(CXXFLAGS)

julia_png_cuda: julia_png_cuda.cu $(LODEPNG)
	$(NVCC) -o $@ $^

clean:
	rm -f julia_png julia_png_cuda
