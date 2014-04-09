/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include "opencl_utils.hpp"

random_seed* generate_seeds(CLCONTEXT* context, unsigned int number_of_workers)
{
	random_seed *seeds = (random_seed*)clmalloc(context, number_of_workers*sizeof(random_seed), 0);
	for(int i = 0; i < number_of_workers; i++)
	{
		seeds[i].x = rand();
		seeds[i].y = rand();
	}
	
	return seeds;
}
