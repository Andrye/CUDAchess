#include "alphabeta.h"

__global__
void alpha_beta(node* nodes, float *d_values, unsigned int depth, int n_children){
    for(unsigned int i = 0; i < n_children; i++)
        d_values[i] = 0.;
}
