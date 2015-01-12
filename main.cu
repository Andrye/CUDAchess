#include "cuda.h"
#include "chess.h"
#include <cstdint>
#include <cstdio>
#include <vector>

void launchKernel(){
    const int nThreads = 1024;
    Game* games = (Game*) malloc(sizeof(Game) * nThreads);
    for(int i = 0; i < nThreads; i++){
        games[i] = Game();
    }

    Game *devGames;
    Move *devMoves;
    float* devValues;

    cudaMalloc((void**) &devMoves, sizeof(Move) * nThreads);
    cudaMalloc((void**) &devValues, sizeof(float) * nThreads);
    cudaMalloc((void**) &devGames, sizeof(Game) * nThreads);
    cudaMemcpy(devGames, games, sizeof(Game) * nThreads, cudaMemcpyHostToDevice);
    dim3 numThreads(nThreads, 1, 1);

    alphaBeta<<<1, numThreads>>>(devGames, devMoves, devValues);
    cudaDeviceSynchronize();
}



int main(){

    launchKernel();
    printf("Implement me\n");
    return 0;
}
