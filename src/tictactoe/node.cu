#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "node.h"

extern const float INF = 1600000000;
extern const float NODE_INACCESSIBLE = -INF;

__host__
std::ostream &operator <<(std::ostream &os, node const &n){
    for(int x = 0; x < 4; x++) os << "x" << x << ":                  "; os << "\n";
    for(int x = 0; x < 4; x++) os <<"    y0  y1  y2  y3   "; os << "\n";
    char const* h = "  +---+---+---+---+  ";
    os << h << h << h << h << "\n";
    for(int z = 0; z < 4; z++){
        for(int x = 0; x < 4; x++){
            os << "z" << z;
            for(int y = 0; y < 4; y++){
                int bit = x + 4 * y + 16 * z;
                os << "| " << ((n.xs>>bit &1)?'X':((n.os>>bit)&1)?'O':' ') << " ";
            }
            os << "|  ";
        }
        os << "\n" << h << h << h << h << "\n";
    }
    return os;
}

__host__
inline int get_code_hash(std::string const &line)
{
	return line[0] - '0' + 4 * (line[2] - '0') + 16 * (line[4] - '0');
}

__host__
int parse_move(std::string const &line){
    if (line[0] < '0' || '3' < line[0] || line[1] != ' ' || 
        line[2] < '0' || '3' < line[2] || line[3] != ' ' ||
        line[4] < '0' || '3' < line[4] || (line[5] != '\0' && line[5] != '\n'))
        return -1;
    return get_code_hash(line);
}

unsigned int get_console_move(node const &current_node){
    std::cout << current_node << "node value:" << value(current_node) << std::endl;
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
    n->xs = n->os = 0;
}

//Thank you, wikipedia!
const uint64_t m1  = 0x5555555555555555;
const uint64_t m2  = 0x3333333333333333;
const uint64_t m4  = 0x0f0f0f0f0f0f0f0f;
//const uint64_t m8  = 0x00ff00ff00ff00ff;
//const uint64_t m16 = 0x0000ffff0000ffff;
//const uint64_t m32 = 0x00000000ffffffff;
//const uint64_t hff = 0xffffffffffffffff;
const uint64_t h01 = 0x0101010101010101;

__host__ __device__
int popcount(uint64_t x){
    x -= (x >> 1)& m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;
    return (x * h01) >> 56;
}

__host__ __device__
bool get_child(node const& parent, unsigned int id, node *d_child){
    if(((parent.xs | parent.os) >> id) & 1){
        return false;
    }
    if(d_child != nullptr){
        d_child->os = parent.xs;
        d_child->xs = parent.os + (((uint64_t)1) << id);
    }
    return true;
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
    if(stats[4]) return INF;
    if(stats[8]) return -INF;
    float c = 1.;
    float v = 0;
    for(int i = 1; i <= 4; i++){
        v += c * (stats[i] - stats[i + 4]);
        c *= 152;
    }
    return v;
}

__host__
bool is_terminal(node const &n){
    assert((n.xs & n.os) == 0);
    if(popcount(n.xs | n.os) == 64) return true;
    int stats[9] = {};
    line_stats(n, stats);
    return stats[4] || stats[8];
}
