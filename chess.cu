#include "cuda.h"
#include "chess.h"



__host__ __device__ 
Board::Board(){
    for(File f = fileA;f <= fileH; ++f)
        for(Rank r = rank1; r <= rank8; ++r)
            pieces[f][r] = {PieceType::none, Player::white};
}

__host__ __device__
Board Board::initialPosition(){
    PieceType backRank[] = {
        PieceType::rook, 
        PieceType::knight, 
        PieceType::bishop, 
        PieceType::queen, 
        PieceType::king, 
        PieceType::bishop, 
        PieceType::knight,
        PieceType::rook
    };
    Board b;
    for(File f = fileA; f <= fileH; ++f){
        b.pieces[f][rank1] = {backRank[f], Player::white};
        b.pieces[f][rank2] = {PieceType::pawn, Player::white};
        for(Rank r = rank3; r <= rank6; ++r)
            b.pieces[f][r] = Piece();
        b.pieces[f][rank7] = {PieceType::pawn, Player::black};
        b.pieces[f][rank8] = {backRank[f], Player::black};
    }
    return b;
}

Piece &Board::operator()(File f, Rank r){
    return pieces[f][r];
}
