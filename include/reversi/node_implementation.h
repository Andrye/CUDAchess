#ifndef NODE_IMPLENTATION_H_INCLUDED
#define NODE_IMPLENTATION_H_INCLUDED

enum field_state {
    EMPTY,
    WHITE ,
    BLACK = 2
};

struct node{
    uint64_t xs;
    uint64_t os;
    uint16_t board[8*8];
};

#define N_CHILDREN 64
#endif
