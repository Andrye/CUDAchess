#include "cuda.h"

#include "alphabeta.h"
#include "node.h"
#include <cstdint>
#include <cstdio>
#include <vector>

const int DEPTH = 5;
extern const int n_children;

void get_players_move(node * n, node * next_node);
void get_bots_move(node * n, node * next_node);


node launchKernel(node * current_node){
    const int n_threads = 1024;

    node* nodes =  allocate_nodes(n_threads);
    for(int i = 0; i < n_threads; i++){
        init_node(ptr_plus(nodes, i));
    }

    node *dev_nodes;
    float* dev_values;

    cudaMalloc((void**) &dev_values, sizeof(float) * n_threads);
    cudaMalloc((void**) &dev_nodes, node_size() * n_threads); //TODO: so far I decided that it indeed should be n_threads, not n_blocks. We'll see later.
    cudaMemcpy(dev_nodes, nodes, node_size() * n_threads, cudaMemcpyHostToDevice);
    dim3 numThreads(n_threads, n_children, 1);
	
	node best_move;
    alpha_beta(dev_nodes, dev_values, current_node, DEPTH, n_children, &best_move, numThreads); //TODO: for now it's cudaDeviceSynchronize();
	//printf("%d\n", best_move.xs + best_move.os);
	return best_move;
}



int main(){
    node *nodes = allocate_nodes(2);
    init_node(nodes);
    node *n = nodes;
    for(int i = 0; !is_terminal(ptr_plus(nodes, i)); i=1-i, n = ptr_plus(nodes, i)){
        if(i==0)
			get_players_move(n, ptr_plus(nodes, 1-i));
		else
			get_bots_move(n, ptr_plus(nodes, 1-i));
	}
    printf("Implement me\n");
    return 0;
}


void get_players_move(node * my_node, node * next_node)
{
		        
	print_node(my_node);
	printf("node value: %f\n", value(my_node));
	int move = -1;
	do {
		printf("your move\n");
		char *lineptr = nullptr;
		size_t len = 0;
		getline(&lineptr, &len,  stdin);
		move = scan_move(lineptr);
		free(lineptr);
	} while(move < 0 || !get_child(my_node, move, next_node));
}

void get_bots_move(node * n, node * next_node)
{
	* next_node = launchKernel(n);
}


            
            
