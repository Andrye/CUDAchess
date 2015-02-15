CUDA_INSTALL_PATH ?= /usr/local/cuda

NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Options
NVCCOPTIONS = -arch sm_20 -std=c++11 -rdc=true -g -G

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)



.SUFFIXES:	.cu	

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -Itictactoe -c $< -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCOPTIONS) $(OBJS) main.cu -o $(TARGET)

tictactoe: alphabeta.cu.o tictactoe.cu.o main.cu.o
	$(NVCC) $(NVCCOPTIONS) tictactoe.cu.o alphabeta.cu.o main.cu.o -o tictactoe.e

clean:
	rm -rf *.x *.o


