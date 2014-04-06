/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <iostream>

# include <cstdio>
# include <cstdlib>
# include <ctime>

# include <stdcl.h>

int main(int argc, char **argv)
{
	srand(time(NULL));

	stdcl_init();

	/* use default contexts, if no GPU use CPU */
	CLCONTEXT* context = (stdgpu)? stdgpu : stdcpu;

	unsigned int devnum = 0;

	clopen(context, NULL, CLLD_NOW);
	
	cl_kernel kernel = clsym(context, NULL, "random_kernel", 0);
	if (!kernel)
	{
		std::cerr << "error: kernel = " << kernel << std::endl;
		return 1;
	}

	unsigned int work_size = 10000;
	unsigned int workers = 16;

	cl_uint *seed1 = (unsigned int*)clmalloc(context, 4*workers*sizeof(cl_uint), 0);
	cl_uint *seed2 = (unsigned int*)clmalloc(context, 4*workers*sizeof(cl_uint), 0);
	for(int i = 0; i < 4*workers; i++)
	{
		seed1[i] = rand();
		seed2[i] = rand();
	}

	clmsync(context, devnum, seed1, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, devnum, seed2, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	
	/* allocate OpenCL device-sharable memory */
	cl_float* mean = (float*)clmalloc(context, workers*sizeof(cl_float), 0);
	cl_float* variance = (float*)clmalloc(context, workers*sizeof(cl_float), 0);
	
	for(int i = 0; i < workers; i++)
	{
		mean[i] = 23.0f;
		variance[i] = 42.0f;
	}

	/* define the computational domain and workgroup size */
	clndrange_t index_range = clndrange_init1d(0, workers, 2);

	/* non-blocking fork of the OpenCL kernel to execute on the GPU */
	clforka(context, devnum, kernel, &index_range, CL_EVENT_NOWAIT, work_size, seed1, seed2, mean, variance);

	/* non-blocking sync vector c to host memory (copy back to host) */
	clmsync(context, devnum,  mean, CL_MEM_HOST|CL_EVENT_NOWAIT);
	clmsync(context, devnum,  variance, CL_MEM_HOST|CL_EVENT_NOWAIT);

	/* force execution of operations in command queue (non-blocking call) */
	clflush(context, devnum, 0);

	/* block on completion of operations in command queue */
	clwait(context, devnum, CL_ALL_EVENT);

	for(int i = 0; i < workers; i++)
	{
		printf("Worker %d:\n\tMean:     %10.5f\n\tVariance: %10.5f\n", i, mean[i], variance[i]);
	}

	clfree(seed1);
	clfree(seed2);
	clfree(mean);
	clfree(variance);

	return 0;
}
