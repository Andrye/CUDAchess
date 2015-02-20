#ifndef NODE_H_INCLUDED
#define NODE_H_INCLUDED
#include<ostream>


struct node;

#include "node_implementation.h"

#ifndef N_CHILDREN
#error N_CHILDREN not defined in node_implementation.h
#endif

std::ostream &operator<<(std::ostream &osn, node const&);


unsigned int get_console_move(node const&);

__host__ __device__
bool get_child(node const&, unsigned int, node*);
__host__ __device__
bool get_child(node *, unsigned int, node*);

__host__ __device__
float value(node const&);

__host__ __device__
bool is_terminal(node const&);

#endif
