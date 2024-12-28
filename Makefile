NVCC = nvcc
CXX = g++
CUDAFLAGS = -arch=sm_87


SRCS = ./src/*.cu 
OBJS = $(SRCS:.cu=.o)

TARGET = perf

all: $(TARGET)

%.o: %.cu
	$(NVCC) -O3 -std=c++11 $(CUDAFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@ 
clean:
	rm -rf *.o $(TARGET)
	

