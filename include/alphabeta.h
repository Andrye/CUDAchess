#ifndef ALPHABETA_H_INCLUDED
#define ALPHABETA_H_INCLUDED
#include "node.h"


__host__
float alpha_beta(node * nodes, float * d_values, node *current_node, unsigned int depth, int n_children, node * best_move, dim3 numThreads);

__global__
void compute_other_nodes (node *nodes, float *dev_values, node *current_node, unsigned int depth, int n_children, float limit);

__device__
float compute_node(node *current_node, unsigned int depth, int n_children, float limit);

__host__ //global in the future
int get_best_index(float * d_values, int n_children);



__host__ __device__
float invert_limit(float limit);
#endif
