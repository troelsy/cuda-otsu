# Makefile for cuda-otsu project

# Compiler and flags
CXX := g++
NVCC := nvcc
CUDA_HOME := /usr/local/cuda
CXXFLAGS := -O2 -std=c++11 -I$(CUDA_HOME)/include
NVCCFLAGS := -O2 -arch=sm_80 -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -lcudart -Xcompiler -fPIC

# Source files
SRCS := example.cc otsu.cu
OBJS := $(SRCS:.cc=.o)
OBJS := $(OBJS:.cu=.o)

# Executable name
TARGET := example

# Default rule
all: $(TARGET)

# Link objects to create the executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile C++ source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
