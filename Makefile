CUDA_INSTALL_PATH ?= /usr/local/cuda

NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc
INCLUDES = -Iinclude -I$(CUDA_INSTALL_PATH)/include

# Options
NVCCOPTIONS = -arch sm_20 -std=c++11 -rdc=true -g -G

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)



.SUFFIXES:	.cu	

help:
	@echo "CUDA alfa beta game project"
	@echo "To build a game enter : "
	@echo ""
	@echo "\tmake <game name>"
	@echo ""
	@echo "Currently implemented games : "
	@echo "\t- tictactoe"
	@echo ""
	@echo "To run a game enter : ./<game name>.x"
	@echo ""

%.main.cu.o: src/main.cu
	$(NVCC) $(NVCCFLAGS) -Iinclude/$* -c $< -o $@

%.alphabeta.cu.o: src/alphabeta.cu
	$(NVCC) $(NVCCFLAGS) -Iinclude/$* -c $< -o $@

%.node.cu.o: src/%/node.cu
	$(NVCC) $(NVCCFLAGS) -Iinclude/$* -c $< -o $@

%:  %.main.cu.o %.alphabeta.cu.o %.node.cu.o
	$(NVCC) $(NVCCFLAGS) $*.node.cu.o $*.alphabeta.cu.o $*.main.cu.o -o $@.x

.PHONY: help clean

clean:
	rm -rf *.x *.o


