#include "cuda.h"
#include "chess.h"

__host__ __device__ 
Game::Game(){
    for(File f = fileA;f <= fileH; ++f)
        for(Rank r = rank1; r <= rank8; ++r)
            pieces[f][r] = {PieceType::none, Player::white};
    currentPlayer = Player::white;
}

__host__ __device__
Game Game::initialPosition(){
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
    Game g;
    for(File f = fileA; f <= fileH; ++f){
        g.pieces[f][rank1] = {backRank[f], Player::white};
        g.pieces[f][rank2] = {PieceType::pawn, Player::white};
        g.pieces[f][rank7] = {PieceType::pawn, Player::black};
        g.pieces[f][rank8] = {backRank[f], Player::black};
    }
    return g;
}

Piece const &Game::operator()(File f, Rank r) const {
    return pieces[f][r];
}

__host__ __device__
int Game::generateMoves(Move* moves, int maxMoves, int skip) const{
    *moves = {{fileE, rank2}, {fileE, rank4}, PieceType::none};
    return 1;
}


__host__ __device__
float Game::gameValue(){
    return 0.;
}
__host__ __device__
void Game::applyMove(Move m){
    pieces[m.to.file][m.to.rank] = pieces[m.from.file][m.from.rank];
    pieces[m.from.file][m.from.rank] = Piece();
}

__global__
void alphaBeta(Game *games, Move *moves, float *values){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    Move move;
    int nMoves = games[thid].generateMoves(&move, 1);
    games[thid].applyMove(move);
    moves[thid] = move;
    values[thid] = games[thid].gameValue();

}
