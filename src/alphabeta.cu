#include "alphabeta.h"
#include <stdio.h>


/********** Note that, unlike theearly versions, every funtion returns the value of node X
            for the player who is on the move in node X. It it a calling function's responsibility to invert
            the value. This should only be changed consistently everywhere in this file.
***********/

//extern const float INF, NODE_INACCESSIBLE;

const float INF = 1600000000;
const float NODE_INACCESSIBLE = -INF;
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


__global__
void compute_children_of_a_node (node *nodes, float *dev_values, node * current_node, unsigned int depth, int n_children, AB limit);

__device__
float compute_node(node const &current_node, unsigned int depth, int n_children, AB limit);

__host__
int get_best_index(float * d_values, int n_children);


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
float alpha_beta(node * nodes, float * d_values, node const &current_node, unsigned int depth, int n_children, unsigned int * best_move_value, dim3 numThreads) //TODO: for now it's assumed n_children < legth of "nodes" array
{
    if(depth == 0 || is_terminal(current_node))
	{
		return value(current_node);
	}


	node child;
	
	int __index_of_recursive_estimation; //this variable can probably be deleted in the final version, but is crucial untill GPU has recursion as well as CPU
	for (int i=0; i<n_children; i++)     //it should be sort, shouldn't it?
        if(get_child(current_node, i, &child))
	    {
	        __index_of_recursive_estimation = i;
	        break;
	    }

	float limit_estimation = invert( alpha_beta(nodes, d_values, child, depth - 1, n_children, nullptr, numThreads) );
    
    float * values = new float[n_children]; //it'll be done better in the future,
    
    node * dev_current_node;
    cudaMalloc((void**) &dev_current_node, sizeof(node));
    cudaError_t cudaResult = cudaMemcpy(dev_current_node, &current_node, sizeof(node), cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        printf("cuda error  %d\n", cudaResult);
        throw 1;
    }

	compute_children_of_a_node <<<1, numThreads>>> (nodes, d_values, dev_current_node, depth, n_children, AB(limit_estimation, INF));
    cudaResult = cudaDeviceSynchronize();
    if(cudaResult != cudaSuccess)
    {
        printf("cuda error %s\n", cudaGetErrorString(cudaResult));
        throw 1;
    }
    
    cudaResult = cudaMemcpy(values, d_values, sizeof(float) * n_children, cudaMemcpyDeviceToHost);
    if(cudaResult != cudaSuccess)
    {
        printf("cuda error  %d\n", cudaResult);
        throw 1;
    }
	int best_ind = get_best_index(values, n_children);
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

	//float best_val = thrust::reduce(d_values, d_values + n_children, thrust::maximum<float>); //TODO: can we use library magic or should we paste out code for scan?    
   
}

__global__
void compute_children_of_a_node (node *nodes, float *values, node * current_node, unsigned int depth, int n_children, AB limit)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    //int bl_id = blockIdx.x; //TODO: no blocks so far
    //int th_id = threadIdx.x;
    
    node child;
    node * childptr = &child;
    
    if(get_child(*current_node, id, childptr))
        values[id] = invert( compute_node(child, depth - 1, n_children, invert(limit)) );
    else
        values[id] = NODE_INACCESSIBLE;
}

__device__
float compute_node(node const &current_node, unsigned int depth, int n_children, AB limit)
{
	if(is_terminal(current_node))
		return value(current_node);
	
	node child;
	float best_res = -INF;
	
	for (int i=0; i<n_children; i++)
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
int get_best_index(float * values, int n_children)
{
	int res_index;
	float val = -1e10;
	for (int i=0; i<n_children; i++)
	{
		if(values[i] > val)
		{
			val = values[i];
			res_index = i;
		}
	}
	return res_index;
}

