#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include "node.h"

const float INF = std::numeric_limits<float>::infinity();


__host__ __device__
uint64_t bit_idx(int x, int y){
    return ((uint64_t)1)<<(8*y + x);
}

__host__
std::ostream &operator <<(std::ostream &os, node const &n){
  bool oplayer = (__builtin_popcountll(n.xs | n.os) % 2 == 0);
  char O = oplayer ? 'O' : '@';
  char X = 'O' ^ '@' ^ O;
    char const* h = "+---+---+---+---";
    os << h << h << "+\n";
    for(int z = 0; z < 8; z++){
        for(int x = 0; x < 8 ; x++) {
          uint64_t bit = bit_idx(x,z);
          os << "| "<< ((n.xs) & bit ? X : ((n.os) & bit ? O : ' ' ))<<' ';
        }
        os << "|\n"<< h << h<< "+\n";
    }
    return os;
}

__host__
inline int get_code_hash(std::string const &line)
{
	return line[0] - '0' + 8 * (line[2] - '0');
}

__host__
int parse_move(std::string const &line){
    if (line[0] < '0' || '7' < line[0] || line[1] != ' ' || 
        line[2] < '0' || '7' < line[2] || (line[3] != '\0' && line[3] != '\n'))
        return -1;
    return get_code_hash(line);
}

unsigned int get_console_move(node const &current_node){
    int move = -1;
    do {
        std::cout << "your move" << std::endl;
        std::string line;
        std::getline(std::cin, line);
        move = parse_move(line);
    } while(move < 0 || !get_child(current_node, move, nullptr));
    return (unsigned int) move;
}

extern const int n_children = 64;

__host__
void init_node(node *n){
    n->xs = 34628173824;
    n->os = 68853694464;
}

const uint64_t m1  = 0x5555555555555555;
const uint64_t m2  = 0x3333333333333333;
const uint64_t m4  = 0x0f0f0f0f0f0f0f0f;
const uint64_t h01 = 0x0101010101010101;

__host__ __device__
int popcount(uint64_t x){
    x -= (x >> 1)& m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;
    return (x * h01) >> 56;
}

__host__ __device__ 
bool good_segment(int pos_x, int pos_y, node const & n, int s_x, int s_y)
{
    int cnt = 0;
    while (0<=pos_x && pos_x<8 && 0<=pos_y && pos_y<8)
    {
        uint64_t bit = bit_idx(pos_x,pos_y);
        if (n.os & bit) return cnt;
        if (!(n.xs & bit)) return false;
        pos_x += s_x;
        pos_y += s_y;
        ++cnt;
    }
    return false;
}

__host__ __device__
void reverse(int pos_x, int pos_y, node & n, int s_x, int s_y)
{
    while (0<=pos_x && pos_x<8 && 0<=pos_y && pos_y<8)
    {
        uint64_t bit = bit_idx(pos_x, pos_y);
        if (n.os & bit) return;
        n.os ^= bit;
        n.xs ^= bit;
        pos_x += s_x;
        pos_y += s_y;
    }
}

__host__ __device__
bool is_valid(unsigned int id, node const & n)
{
    int x = id%8;
    int y = id/8;
    bool is_valid = good_segment(x+1, y+0, n, +1, +0); 
        is_valid |= good_segment(x+1, y+1, n, +1, +1); 
        is_valid |= good_segment(x+0, y+1, n, +0, +1); 
        is_valid |= good_segment(x-1, y+1, n, -1, +1); 
        is_valid |= good_segment(x-1, y+0, n, -1, +0); 
        is_valid |= good_segment(x-1, y-1, n, -1, -1); 
        is_valid |= good_segment(x+0, y-1, n, +0, -1); 
        is_valid |= good_segment(x+1, y-1, n, +1, -1); 
    return is_valid;
}

__host__ __device__
void global_reverse(int x, int y, node & n)
{
    if (good_segment(x+1, y+0, n, +1, +0)) reverse(x+1, y+0, n, +1, +0);
    if (good_segment(x+1, y+1, n, +1, +1)) reverse(x+1, y+1, n, +1, +1);
    if (good_segment(x+0, y+1, n, +0, +1)) reverse(x+0, y+1, n, +0, +1);
    if (good_segment(x-1, y+1, n, -1, +1)) reverse(x-1, y+1, n, -1, +1);
    if (good_segment(x-1, y+0, n, -1, +0)) reverse(x-1, y+0, n, -1, +0);
    if (good_segment(x-1, y-1, n, -1, -1)) reverse(x-1, y-1, n, -1, -1);
    if (good_segment(x+0, y-1, n, +0, -1)) reverse(x+0, y-1, n, +0, -1);
    if (good_segment(x+1, y-1, n, +1, -1)) reverse(x+1, y-1, n, +1, -1);
}

__host__ __device__
bool get_child(node const& parent, unsigned int id, node *d_child){
    assert(!(parent.os & parent.xs));
    int x = id % 8;
    int y = id / 8;
    if(((parent.os | parent.xs) >> id) & 1) return false;
    node tmp;
    tmp.os = parent.os + (((uint64_t)1) << id);
    tmp.xs = parent.xs;
    if (is_valid(id, tmp))
    {
        global_reverse(x, y, tmp);
        if(d_child != nullptr){
            d_child->os = tmp.xs;
            d_child->xs = tmp.os;
	        assert(!(d_child->os & d_child->xs));
        }
        return true;
    }
    return false;
}

__host__ __device__
int count_line(node const *n, uint64_t line){
    int count = 0;
    if((n->xs & line) == (uint64_t) 0) count++;
    if((n->os & line) == (uint64_t) 0) count--;
    return count;
}

uint64_t make_line(int x0, int dx, int y0, int dy, int z0, int dz){
    uint64_t line = 0;
    for(int i = 0; i < 4; i++){
        line = line | (((uint64_t) 1) << (x0 + i*dx) + 4*(y0 + i*dy) + 16*(z0 + i*dz));
    }
    return line;

}

__host__ __device__
int line_type(node const &n, uint64_t line){
    int os = popcount(n.os & line);
    int xs = popcount(n.xs & line);
    if((os == 0) == (xs == 0)) return 0;
    return os ? os : xs + 4;
}
// stats[0] := number of lines with no Os or Xs or with both Os and Xs  //Os = mine, Xs = yours
// stats[i=1..4] := number of lines with i Os and no Xs
// stats[i=5..8] := number of lines with i-4 Xs and no Os;
__host__ __device__
void line_stats(node const &n, int* stats){
    //"1D" lines:
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++){
            stats[line_type(n, 0x000000000000000full << (4 * i + 16 * j))]++;
            stats[line_type(n, 0x0000000000001111ull << (i + 16 * j))]++;
            stats[line_type(n, 0x0001000100010001ull << (i + 4 * j))]++;
        }

    //"2D" lines:
    for(int i = 0; i < 4; i++){
        stats[line_type(n, 0x1000010000100001ull << i)]++;
        stats[line_type(n, 0x0001001001001000ull << i)]++;
        stats[line_type(n, 0x0008000400020001ull << (4 * i))]++;
        stats[line_type(n, 0x0001000200040008ull << (4 * i))]++;
        stats[line_type(n, 0x0000000000008421ull << (16 * i))]++;
        stats[line_type(n, 0x0000000000001248ull << (16 * i))]++;
    }

    //"3D" lines:
    stats[line_type(n, 0x0001002004008000ull)]++;
    stats[line_type(n, 0x0008004002001000ull)]++;
    stats[line_type(n, 0x1000020000400008ull)]++;
    stats[line_type(n, 0x8000040000200001ull)]++;
}


__host__ __device__
float value(node const& n){
    int stats[9] = {};
    line_stats(n, stats);
    float c = 1.;
    float v = 0;
    for(int i = 1; i <= 4; i++){
        v += c * (stats[i] - stats[i + 4]);
        c *= 152;
    }
    return v;
}

__host__ __device__
bool is_terminal(node const &n){
    return (popcount(n.xs | n.os) == 64);
}
