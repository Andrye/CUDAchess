#ifndef ALPHABETA_H_INCLUDED
#define ALPHABETA_H_INCLUDED
#include "node.h"

__global__
void alpha_beta(node* nodes, float *d_values, unsigned int depth, int n_children);
#endif
