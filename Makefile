CUDA_INSTALL_PATH ?= /usr/local/cuda

NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Options
NVCCOPTIONS = -arch sm_20 -std=c++11

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)

OBJS = chess.cu.o
TARGET = chess.x

.SUFFIXES:	.cu	

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCOPTIONS) $(OBJS) main.cu -o $(TARGET)

clean:
	rm -rf $(TARGET) *.o


