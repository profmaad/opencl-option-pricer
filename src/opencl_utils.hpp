/* (c) 2014 Maximilian Gerhard Wolter */

# ifndef OCLOP_OPENCL_UTILS_H
# define OCLOP_OPENCL_UTILS_H

# include <stdcl.h>

typedef cl_uint2 random_seed;

template<typename To, typename From>
To* opencl_memcpy(CLCONTEXT *context, unsigned int size, const From *source)
{
	To *cl_mem = (To*)clmalloc(context, size*sizeof(To), 0);
	
	for(int i = 0; i < size; i++)
	{
		cl_mem[i] = source[i];
	}

	return cl_mem;
}

random_seed* generate_seeds(CLCONTEXT* context, unsigned int number_of_workers);

# endif /*OCLOP_OPENCL_UTILS_H*/
