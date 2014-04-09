/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_OPENCL_UTILS_H
# define OCLOP_OPENCL_UTILS_H

# include <stdcl.h>

template<typename To, typename From>
To* opencl_memcpy(CLCONTEXT *context, unsigned int size, From *source)
{
	To *cl_mem = (To*)clmalloc(context, size*sizeof(To), 0);
	
	for(int i = 0; i < size; i++)
	{
		cl_mem[i] = source[i];
	}

	return cl_mem;
}

# endif /*OCLOP_OPENCL_UTILS_H*/
