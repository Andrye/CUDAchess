#ifndef ALPHABETA_H_INCLUDED
#define ALPHABETA_H_INCLUDED
#include "node.h"


__host__
//float alpha_beta(node * nodes, float * d_values, node const &current_node, unsigned int depth, unsigned int * best_move, dim3 numThreads);
float alpha_beta(node const &current_node, unsigned int depth, unsigned int * best_move_value, dim3 numThreads);

unsigned int get_alpha_beta_gpu_move(node const&);
unsigned int get_alpha_beta_cpu_move(node const&);
unsigned int get_alpha_beta_cpu_kk_move(node const&);

#endif
