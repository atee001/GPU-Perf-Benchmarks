NVCC = nvcc
CXX = g++
CUDAFLAGS = --generate-code arch=compute_87,code=sm_87

INCLUDE_DIR = ./include
CXXFLAGS = -O3 -std=c++11

SRCS = $(wildcard ./src/*.cu)
OBJS = $(SRCS:./src/%.cu=./build/%.o) #src is in src dir put .o in build dir
TARGET = ./build/perf
$(shell mkdir -p ./build)

all: $(TARGET)

./build/%.o: $(SRCS)
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(TARGET): $(OBJS) 
	$(NVCC) $(OBJS) -o $@ 

clean:
	rm -rf ./build/*.o $(TARGET)
	

