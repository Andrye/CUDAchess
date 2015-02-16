#ifndef NODE_H_INCLUDED
#define NODE_H_INCLUDED
#include<ostream>


struct node;

#include "node_implementation.h"


std::ostream &operator<<(std::ostream &osn, node const&);

extern const int n_children;

unsigned int get_console_move(node const&);

__host__ __device__
bool get_child(node const&, unsigned int, node*);

__host__ __device__
float value(node const&);

__host__ __device__
bool is_terminal(node const&);

#endif
