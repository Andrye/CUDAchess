#include "cuda.h"
#include "chess.h"
#include <cstdint>
#include <cstdio>
#include <vector>


int main(){
    Board b;
    b(fileE, rank4) = b(fileE, rank2);
    b(fileE, rank2) = Piece();
    printf("Implement me\n");
}
