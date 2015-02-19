#ifndef ALPHABETA_H_INCLUDED
#define ALPHABETA_H_INCLUDED
#include "node.h"


__host__
float alpha_beta(node * nodes, float * d_values, node const &current_node, unsigned int depth, int n_children, unsigned int * best_move, dim3 numThreads);

#endif
