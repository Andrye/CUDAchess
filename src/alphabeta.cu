#include "alphabeta.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <cstdio>
#include <chrono>

/********** Note that, unlike theearly versions, every funtion returns the value of node X
            for the player who is on the move in node X. It it a calling function's responsibility to invert
            the value. This should only be changed consistently everywhere in this file.
***********/

//extern const float INF, NODE_INACCESSIBLE;

const float INF = std::numeric_limits<float>::infinity();

//TODO: how do we make extern const variables visible on device apart from stupid copying?
//If there is no elegant solution, then macro?

struct AB
{
	float a, b;
    
    __host__ __device__
	AB(){}
    
    __host__ __device__
	AB(float a, float b) : a(a), b(b) {}
};

struct stack_entry {

    AB limits;
    node current_node;
    int idx; // id of child to be searched.
    char valid_children[(N_CHILDREN + 7) / 8]; // bit mask of valid children

};

__global__
void compute_children_of_a_node (node *nodes, float *dev_values, node * current_node, unsigned int depth, AB limit);

__device__
float compute_node(node const &current_node, unsigned int depth, AB limit);

__host__
int get_best_index(float * d_values);


__host__ __device__
float invert(float limit)
{   
	return -limit;
}
__host__ __device__
AB invert (AB val)
{
	return AB( invert(val.b), invert(val.a) );
}


/* nodes - unused till we implementbclocks
 * best_move - can be nullptr if we only want the numerical result
 */
__host__
float alpha_beta(node * nodes, float * d_values, node const &current_node, unsigned int depth, unsigned int * best_move_value, dim3 numThreads) //TODO: for now it's assumed N_CHILDREN < legth of "nodes" array
{
    if(depth == 0 || is_terminal(current_node))
	{
		return value(current_node);
	}


	node child;
	
	int __index_of_recursive_estimation; //this variable can probably be deleted in the final version, but is crucial untill GPU has recursion as well as CPU
	for (int i=0; i<N_CHILDREN; i++)     //it should be sort, shouldn't it?
        if(get_child(current_node, i, &child))
	    {
	        __index_of_recursive_estimation = i;
	        break;
	    }

	float limit_estimation = invert( alpha_beta(nodes, d_values, child, depth - 1, nullptr, numThreads) );
    
    float * values = new float[N_CHILDREN]; //it'll be done better in the future,
    
    node * dev_current_node;
    cudaMalloc((void**) &dev_current_node, sizeof(node));
    cudaError_t cudaResult = cudaMemcpy(dev_current_node, &current_node, sizeof(node), cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        printf("cuda error  %d\n", cudaResult);
        throw 1;
    }

	compute_children_of_a_node <<<1, numThreads>>> (nodes, d_values, dev_current_node, depth, AB(limit_estimation, INF));
    cudaResult = cudaDeviceSynchronize();
    if(cudaResult != cudaSuccess)
    {
        printf("cuda error %s\n", cudaGetErrorString(cudaResult));
        throw 1;
    }
    
    cudaResult = cudaMemcpy(values, d_values, sizeof(float) * N_CHILDREN, cudaMemcpyDeviceToHost);
    if(cudaResult != cudaSuccess)
    {
        printf("cuda error  %d\n", cudaResult);
        throw 1;
    }
	int best_ind = get_best_index(values);
	float result = values[best_ind];
	
    /******** can be deleted in the final version ********/
    if(result <= limit_estimation)
        best_ind = __index_of_recursive_estimation;
            
	/*if(best_move != nullptr)
    {
        //don't look at this code, it's stupid. But I want to finish it now
		cudaMemcpy(best_move, nodes + best_ind, sizeof(node), cudaMemcpyDeviceToHost);
    }*/

	delete[] values;
	cudaFree(dev_current_node);
	if(best_move_value != nullptr)
	    *best_move_value = best_ind;
	
	return result;

	//float best_val = thrust::reduce(d_values, d_values + N_CHILDREN, thrust::maximum<float>); //TODO: can we use library magic or should we paste out code for scan?    
   
}

const int MAX_STACK_SIZE = 10;


#define DEBUG if(0)
__global__ 
void alpha_beta_gpu(node *nodes, float *values, unsigned int depth, AB limits_){

    __shared__ stack_entry stack[MAX_STACK_SIZE];
    __shared__ stack_entry* stacklast;
    __shared__ unsigned int valid_children[N_CHILDREN];
    __shared__ bool toContinue;
    __shared__ float children_values[N_CHILDREN];

    int thid = threadIdx.x;
    int blid = blockIdx.x;
    node local_node;
    float ret;

    valid_children[thid] = 0;

    /*    if(blid == 0 && thid == 0) printf("test\n");
            bool has_child = get_child(nodes[blid], thid, nullptr) ? (thid & 0xf) : 0;
            // we want to calculate bit mask of valid children. First we will find singleton masks
            // and then we will run (bitwise or)-scan.
            valid_children[thid] = has_child ? (1 << (thid & 7)) : 0;
	    if(blid == 0)printf("vc[%d] = %x\n", thid, 0xff & valid_children[thid]);
	    
	    //if(blid == 0) printf("%d %d vs[%d]=%x\n", counter++, blid, thid, valid_children[thid]);
	    __syncthreads();
	    if(blid == 0) printf("thread=%d counter=%d line=%d\n", thid, counter++, __LINE__);

	    if(blid == 0 && thid == 0){
	      for(int i = 0; i < N_CHILDREN; i++){

		printf("vc[%d]=%x ", i, 0xff & valid_children[i]);
	      }
	      printf("\n");
	      }*/
    
    if(thid == 0){

        stack[0].limits = limits_;
        stack[0].current_node = nodes[blid];
        stack[0].idx = 0;

        stacklast = stack;
        DEBUG printf("%d alfabeta(%f,%f,(%lx,%lx)) initial call\n", blid, stacklast->limits.b,stacklast->limits.a,stacklast->current_node.os, stacklast->current_node.xs);
    }
    __syncthreads();
    while(stacklast >= stack){
        if(thid == 0){
            toContinue = false;
	    if(stacklast->current_node.os & stacklast->current_node.xs){
	      DEBUG printf("%d This is wrong %lx %lx\n", blid, stacklast->current_node.os, stacklast->current_node.xs);
	    }
        }

        if(is_terminal(stacklast->current_node)){ // if current node is terminal
	  if(thid == 0){
            DEBUG printf("%d node is terminal\n", blid);
            float val = value(stacklast->current_node);
            DEBUG printf("%d val = %f\n", blid, val);
            ret = val;
	    stacklast--;
            toContinue = true;
	  }
        } else if(stacklast == stack + depth){ // if max depth reached
	  DEBUG printf("%d max depth reached.\n", blid);
            
            if(get_child(stacklast->current_node, thid, &local_node)){ // find values of children
                children_values[thid] = value(local_node);
            } else {
                children_values[thid] = INF;
            }

	    DEBUG {
	      printf("%d children values: ", blid);
	      for(int i = 0; i < N_CHILDREN; i++) printf("%d[%d]:%.0f ", blid, i, children_values[i]);
                printf("\n");
            }
	    __syncthreads();
            
            for(int d = 1; d < N_CHILDREN; d <<= 1){
	      
                if((thid ^ d) < N_CHILDREN){ // find min  of these values
		  __syncthreads();
                    float vald = children_values[thid ^ d];
		    float val = children_values[thid];
		    __syncthreads();
                    if (vald < val)
                        children_values[thid] = vald;
                }
            }

            if(thid == 0){
                ret = -children_values[0];
		stacklast--;
                DEBUG printf("%d min = %f (return)\n", blid, ret);
		toContinue = true;
	    }
        }

        __syncthreads();


        if(toContinue){
            continue;
        }
        
        if(stacklast->idx == 0){ // if this is the first time we are at current node
	  DEBUG printf("%d first time in the node\n", blid);
            bool has_child = get_child(stacklast->current_node, thid, nullptr) ? (thid & 0xf) : 0;
            // we want to calculate bit mask of valid children. First we will find singleton masks
            // and then we will run (bitwise or)-scan.
            valid_children[thid] = has_child ? (1 << (thid & 7)) : 0;
	    
	    //if(blid == 0) printf("%d %d vs[%d]=%x\n", counter++, blid, thid, valid_children[thid]);
	    __syncthreads();

 

            //note that no syncthreading is needed below - communication is within warp;
            for(int d = 1; d < 8; d <<= 1){
	      if((thid ^ d) < N_CHILDREN){
                    valid_children[thid] = valid_children[thid] | valid_children[thid ^ d];
	      }
            }

	    __syncthreads();

            if((thid & 7) == 0){
                stacklast->valid_children[thid >> 3] = valid_children[thid];
	    }

	    __syncthreads();

	    DEBUG {
	      for(int i = 0; i < (N_CHILDREN + 7) / 8; i++) printf("%x", 0xff & stacklast->valid_children[i]);
	      printf("\n");
	    }
	    __syncthreads();
        } else { // we've just returned from recursion.

	  DEBUG printf("%d just returned from recursion\n", blid);
            if(thid == 0 && -ret > stacklast->limits.a){
                stacklast->limits.a = -ret;
		DEBUG printf("%d now alpha = %f\n", blid, stacklast->limits.a);
            }
            if(stacklast->limits.a >= stacklast->limits.b){ //pruning
                stacklast->idx = N_CHILDREN;
            }

        }

        __syncthreads();


        if(thid == 0){
            int idx = stacklast->idx;
            //find valid idx
            while(idx < N_CHILDREN && ((stacklast->valid_children[idx >> 3] >> (idx & 7)) & 1) == 0){
	      
                idx++;
	    }


            if(idx == N_CHILDREN){ // if all children searched - return from recursion
	      DEBUG printf("%d return %f\n", blid, stacklast->limits.a);
                ret = stacklast->limits.a;
                stacklast--;
            } else { // otherwise search children.
                
                (stacklast+1)->limits.a = -stacklast->limits.b;
                (stacklast+1)->limits.b = -stacklast->limits.a;
                get_child(stacklast->current_node, idx, &(stacklast+1)->current_node);
                (stacklast+1)->idx = 0;

                stacklast->idx = ++idx;
                stacklast++;
                DEBUG printf("%d alfabeta(%f,%f,(%lx,%lx))\n", blid, stacklast->limits.b,stacklast->limits.a,stacklast->current_node.os, stacklast->current_node.xs);
            }
        }
        __syncthreads();
    }
    if(thid == 0)
      values[blid] = ret;
}

__host__ __device__
float alpha_beta_cpu(node const& n, unsigned int depth, AB limits){
  node c;
  if(is_terminal(n))
    return value(n);
  if(depth == 0){
    float min_val = INF;
    for(int i = 0; i < N_CHILDREN; i++){
      if(get_child(n, i, &c)){
	float val = value(c);
	if(val < min_val)
	  min_val = val;
      }
    }
    return -min_val;
  }
  float best_val = -INF;
  for(int i = 0; i < N_CHILDREN; i++){
    if(get_child(n, i, &c)){
      float val = -alpha_beta_cpu(c, depth-1, AB(-limits.b, -limits.a));
      if(val > best_val)
	best_val = val;
      if(val > limits.a)
	limits.a = val;
      if(limits.a >= limits.b)
	break;
    }
  }
  return best_val;
}

unsigned int get_alpha_beta_gpu_move(node const &n){
    const int depth = 2;
    unsigned int moves[N_CHILDREN];
    node nodes[N_CHILDREN];
    int children_cnt = 0;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
 
    for(unsigned int i = 0; i < N_CHILDREN; i++){
        if(get_child(n, i, &nodes[children_cnt]))
            moves[children_cnt++] = i;
    }

    node* dev_nodes;
    float* dev_values;
    cudaMalloc((void**) &dev_nodes, sizeof(node) * children_cnt);
    cudaMalloc((void**) &dev_values, sizeof(float) * children_cnt);
    cudaMemcpy(dev_nodes, nodes, sizeof(node) * children_cnt, cudaMemcpyHostToDevice);
    dim3 num_threads(N_CHILDREN, 1, 1);
    alpha_beta_gpu<<<children_cnt, num_threads>>>(dev_nodes, dev_values, depth, AB(-INF, INF));
    float values[children_cnt];
    cudaMemcpy(values, dev_values, sizeof(float) * children_cnt, cudaMemcpyDeviceToHost);
    cudaFree((void**) &dev_values);
    cudaFree((void**) &dev_nodes);
    int best = std::min_element(values, values + children_cnt) - values;

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "GPU generation time : " << elapsed_seconds.count() << "s\n";
    return moves[best];
}
unsigned int get_alpha_beta_cpu_move(node const &n){
    const int depth = 2;
    unsigned int moves[N_CHILDREN];
    node nodes[N_CHILDREN];
    int children_cnt = 0;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    for(unsigned int i = 0; i < N_CHILDREN; i++){
        if(get_child(n, i, &nodes[children_cnt]))
            moves[children_cnt++] = i;
    }
    float values[children_cnt];
    AB ab(-INF, INF);
    for(int i = 0; i < children_cnt; i++){
      values[i] = alpha_beta_cpu(nodes[i], depth, ab);
    }
    int best = std::min_element(values, values + children_cnt) - values;

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "CPU generation time : " << elapsed_seconds.count() << "s\n";
    return moves[best];
}

__global__
void compute_children_of_a_node (node *nodes, float *values, node * current_node, unsigned int depth, AB limit)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    //int bl_id = blockIdx.x; //TODO: no blocks so far
    //int th_id = threadIdx.x;
    
    node child;
    node * childptr = &child;
    
    if(get_child(*current_node, id, childptr))
        values[id] = invert( compute_node(child, depth - 1, invert(limit)) );
    else
      values[id] =  -INF;
}

__device__
float compute_node(node const &current_node, unsigned int depth, AB limit)
{
	if(is_terminal(current_node))
		return value(current_node);
	
	node child;
	float best_res = -INF;
	
	for (int i=0; i<N_CHILDREN; i++)
	{
		if(!get_child(current_node, i, &child))
		    continue;
		float temp_res = invert(value(child)); //recursion here should be
		
		if(temp_res > best_res)
		{
			best_res = temp_res;
			if(temp_res > limit.a)
			{
				if(limit.a >= limit.b)
					return INF; 	//alpha-beta prunning - out move is so greate we know B doesn't want
				                    //the parent node. We return INF though in fact best_res should be enough?
	                                //EDIT friday morning. Now I think it should be only best_res, not INF. I'll reconsider it.

	        }
		}
	}
	return best_res;
}
		
    
//we'll paste code from C instead of this function, but will it be any imporvement at all?
__host__
int get_best_index(float * values)
{
	int res_index;
	float val = -1e10;
	for (int i=0; i<N_CHILDREN; i++)
	{
		if(values[i] > val)
		{
			val = values[i];
			res_index = i;
		}
	}
	return res_index;
}

