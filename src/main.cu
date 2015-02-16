#include "cuda.h"

#include "alphabeta.h"
#include "node.h"
#include <cstdint>
#include <iostream>
#include <vector>

const int DEPTH = 5;

unsigned int get_bots_move(node const&);


node launchKernel(node const& current_node){
    const int n_threads = 1024;

    node* nodes = new node[n_threads];
    for(int i = 0; i < n_threads; i++){
	    nodes[i] = {};
    }

    node *dev_nodes;
    float* dev_values;

    cudaMalloc((void**) &dev_values, sizeof(float) * n_threads);
    cudaMalloc((void**) &dev_nodes, sizeof(node) * n_threads); //TODO: so far I decided that it indeed should be n_threads, not n_blocks. We'll see later.
    cudaMemcpy(dev_nodes, nodes, sizeof(node) * n_threads, cudaMemcpyHostToDevice);
    dim3 numThreads(n_threads, n_children, 1);

    node best_move;
    alpha_beta(dev_nodes, dev_values, current_node, DEPTH, n_children, &best_move, numThreads); //TODO: for now it's cudaDeviceSynchronize();
    delete[] nodes;
    return best_move;
}

int main(){
    node nodes[2];
    nodes[0] = {};
    for(int i = 0; !is_terminal(nodes[i]); i=1-i){
        unsigned int move;
	    if(i==0)
	        move = get_console_move(nodes[i]);
	    else
	        move = get_bots_move(nodes[i]);
	    if(!get_child(nodes[i], move, nodes+1-i))
	        throw "Wrong move returned";
    }
    printf("Implement me\n");
    return 0;
}



unsigned int get_bots_move(node const &n)
{
    node next_node = launchKernel(n);
    for(unsigned int i = 0; i < n_children; i++)
        if(get_child(n, i, nullptr))
            return i;
    throw "no move can be done";
}
