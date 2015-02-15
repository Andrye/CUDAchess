#ifndef NODE_H_INCLUDED
#define NODE_H_INCLUDED
#include<cstdint>

#include "node_implementation.h"

__host__
node* allocate_nodes(int);

__host__ __device__
node *ptr_plus(node*, int);

__host__
node const *ptr_plus(node const*, int);

__host__
void print_node(node const*);

__host__
int scan_move(char const*);


extern const int n_children;

__host__
int node_size();

__host__
void init_node(node*);

__host__ __device__
bool get_child(node const*, unsigned int, node*);

__host__ __device__
float value(node const*);

__host__ __device__
bool is_terminal(node const*);

#endif
