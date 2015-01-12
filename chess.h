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
    Piece():type(), player(){};
    __host__ __device__
    Piece(PieceType type, Player player):type(type), player(player){};
};

struct Square {
    File file;
    Rank rank;
    __host__ __device__
    Square():file(), rank(){};
    __host__ __device__
    Square(File file, Rank rank):file(file), rank(rank){};
};

struct Move {
    Square from;
    Square to;
    PieceType captured;
    __host__ __device__
    Move():from(), to(), captured(){};
    __host__ __device__
    Move(Square from, Square to, PieceType captured):from(from), to(to), captured(captured){};
};



class Game {
public:
    __host__ __device__ 
    Game();
    __host__ __device__ 
    static Game initialPosition();
    __host__ __device__ 
    Piece const &operator()(File, Rank) const;

    /**
     * Generates possible moves to range of memory that starts with `dest`.
     *
     * returns number of generated moves.
     *
     * If there are more than `maxMoves` possibles moves, then only `maxMoves` moves are
     * generated. If `skip` is greater than zero, then first `skip` moves are skipped. This
     * allows generating more than `maxMoves` moves by subseqently calling generateMoves.
     */
    __host__ __device__
    int generateMoves(Move* dest, int maxMoves, int skip = 0) const;

    /**
     * Checks if move `m` is a legal one.
     *
     * Returns true if `m` is a legal move, or false otherwise.
     */
    __host__ __device__
    bool isLegalMove(Move m) const;

    /**
     * Applies the move `m`.
     *
     * Behavior for impossible move is undefined.
     */
    __host__ __device__
    void applyMove(Move m);

    /**
     * Undoes the move `m`.
     *
     * Behavior for not undoable move is undefined.
     */
    __host__ __device__
    void undoMove(Move m);

    /**
     * Prints current state of the game to stdout.
     */
    __host__ __device__
    void print() const;

    /**
     * Returns current player.
     */
    __host__ __device__
    Player getCurrentPlayer() const;

    /**
     * Returns heuristic value of game.
     */
    __host__ __device__
    float gameValue();
private:
    Player currentPlayer;
    Piece pieces[8][8];
};


enum class ParseResult {
    ok,
    parseError,
    illegalMove
};

/**
 * Parses move in algebraic notation.
 *
 * If line is a legal move in algebraic notation, puts parsed move to `move` and returns `ok`;
 * otherwise returns ParseResult with information, what went wrong;
 */

ParseResult parseMove(Move* move, Game const& game, const char* line);

/**
 * Performs parallel alpha-beta pruning for games.
 *
 * Puts best move for each game in `games` to `moves` array, and value of this move to `values` array
 */
__global__
void alphaBeta(Game* games, Move* moves, float* values);

#endif
