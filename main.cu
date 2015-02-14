#include "cuda.h"

#include "alphabeta.h"
#include "node.h"
#include <cstdint>
#include <cstdio>
#include <vector>

void launchKernel(){
    const int n_threads = 1024;
    node* nodes =  allocate_nodes(n_threads);
    for(int i = 0; i < n_threads; i++){
        init_node(ptr_plus(nodes, i));
    }

    node *dev_nodes;
    float* dev_values;

    cudaMalloc((void**) &dev_values, sizeof(float) * n_threads);
    cudaMalloc((void**) &dev_nodes, node_size() * n_threads);
    cudaMemcpy(dev_nodes, nodes, node_size() * n_threads, cudaMemcpyHostToDevice);
    dim3 numThreads(n_threads, n_children, 1);

    alpha_beta<<<1, numThreads>>>(dev_nodes, dev_values, 9, n_children);
    cudaDeviceSynchronize();
}



int main(){
    node *nodes = allocate_nodes(2);
    init_node(nodes);
    node *n = nodes;
    for(int i = 0; !is_terminal(ptr_plus(nodes, i)); i=1-i, n = ptr_plus(nodes, i)){
        print_node(n);
        printf("node value: %f\n", value(n));
        int move = -1;
        do {
            printf("your move\n");
            char *lineptr = nullptr;
            size_t len = 0;
            getline(&lineptr, &len,  stdin);
            move = scan_move(lineptr);
            free(lineptr);
        } while(move < 0 || !get_child(n, move, ptr_plus(nodes, 1-i)));
    }
    launchKernel();
    printf("Implement me\n");
    return 0;
}
