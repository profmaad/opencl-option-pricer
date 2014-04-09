/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include "opencl_utils.hpp"

cl_uint2* generate_seeds(CLCONTEXT* context, unsigned int number_of_workers)
{
	cl_uint2 *seeds = (cl_uint2*)clmalloc(context, number_of_workers*sizeof(cl_uint2), 0);
	for(int i = 0; i < number_of_workers; i++)
	{
		seeds[i].x = rand();
		seeds[i].y = rand();
	}
	
	return seeds;
}
