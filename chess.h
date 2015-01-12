#ifndef CHESS_H_INCLUDED
#define CHESS_H_INCLUDED
#include <cstdint>
#include <type_traits>

enum File {
    fileA = 0, fileB = 1, fileC = 2, fileD = 3, fileE = 4, fileF = 5, fileG = 6, fileH = 7
};

__host__ __device__
inline File &operator++(File &f){
    return f = (File) ((int) f + 1);
}

enum Rank {
    rank1 = 0, rank2 = 1, rank3 = 2, rank4 = 3, rank5 = 4, rank6 = 5, rank7 = 6, rank8 = 7
};

__host__ __device__
inline Rank &operator++(Rank &r){
    return r = (Rank) ((int) r + 1);
}

enum class PieceType {
    none, pawn, knight, bishop, rook, queen, king
};

enum class Player {
    white, black
};

struct Piece {
    PieceType type;
    Player player;
    __host__ __device__
    Piece():type(PieceType::none), player(Player::white){};
    __host__ __device__
    Piece(PieceType type, Player player):type(type), player(player){};
};

class Board {
public:
    __host__ __device__ 
    Board();
    __host__ __device__ 
    static Board initialPosition();
    __host__ __device__ 
    Piece &operator()(File, Rank);
private:
    Piece pieces[8][8];
};

#endif
