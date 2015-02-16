#include "alphabeta.h"
#include <stdio.h>

__global__
void compute_other_nodes (node *nodes, float *dev_values, node const &current_node, unsigned int depth, int n_children, float limit);

__device__
float compute_node(node const &current_node, unsigned int depth, int n_children, float limit);

__host__ //global in the future
int get_best_index(float * d_values, int n_children);



__host__ __device__
float invert_limit(float limit);

/* nodes - for now it's used as a free space for compute_other_nodes(). Maybe it'll change later.
 * depth - not fully implemented as of yet
 * best_move - can be nullptr if we only want the numerical result
 */
__host__ //__global__ 
float alpha_beta(node * nodes, float * d_values, node const &current_node, unsigned int depth, int n_children, node * best_move, dim3 numThreads) //TODO: for now it's assumed n_children < legth of "nodes" array
{
    if(depth == 0)
		return invert_limit(value(current_node));


	node child;
	get_child(current_node, 0, &child); //TODO: not 0, but what? random? some heura to find a sound candidate?
	float limit_estimation = alpha_beta(nodes, d_values, child, depth - 1, n_children, nullptr, numThreads);
	

	compute_other_nodes <<<1, numThreads>>> (nodes, d_values, current_node, depth, n_children, limit_estimation);
    
    float * values = new float[n_children]; //it'll be done better in the future,
        //i.e. taking the best will be on GPU
    cudaMemcpy(values, d_values, sizeof(float) * n_children, cudaMemcpyDeviceToHost);
	
	int best_ind = get_best_index(values, n_children);
	float result = values[best_ind];
	if(best_move != nullptr)
    {
        //don't look at this code, it's stupid. But I want to finish it now
		cudaMemcpy(best_move, nodes + best_ind, sizeof(node), cudaMemcpyDeviceToHost);
	//    printf("  %d\n", (int)(best_move->xs + best_move->os));
    }

	delete[] values;
	
	return invert_limit(result);

	//float best_val = thrust::reduce(d_values, d_values + n_children, thrust::maximum<float>); //TODO: can we use library magic or should we paste out code for scan?    
   
}


__global__
void compute_other_nodes (node *nodes, float *d_values, node const &current_node, unsigned int depth, int n_children, float limit)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    //int bl_id = blockIdx.x; //TODO: no blocks so far
    //int th_id = threadIdx.x;
    
    node child;
    get_child(current_node, id, &child);

	
	compute_node(child, depth - 1, n_children, invert_limit(limit));
	
}

__device__
float compute_node(node const &current_node, unsigned int depth, int n_children, float limit)
{
	float estimate_val = value(current_node);
	if(estimate_val < limit)
		return limit - 1; //because we don't want to watch ourselves to always write < instead of <=, or do we?
	if(is_terminal(current_node))
		return estimate_val;
	
	node child;
	float best_res = limit - 1;
	
	for (int i=0; i<n_children; i++)
	{
		get_child(current_node, i, &child);
		float temp_res = value(child);
		if(temp_res > best_res)
			best_res = temp_res;
	}
	return best_res;
}
		
    
//maybe we'll paste code from C instead of this function, but will it be any imporvement at all?
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


__host__ __device__
float invert_limit(float limit)
{
	return -limit;
}
