/* (c) 2014 Maximilian Gerhard Wolter */

# include <iostream>

# include <cstdio>
# include <cstdlib>
# include <ctime>

# include <stdcl.h>

# include "statistics_opencl.h"

int main(int argc, char **argv)
{
	srand(time(NULL));

	stdcl_init();

	/* use default contexts, if no epiphany use CPU */
	CLCONTEXT* context = (stdacc)? stdacc : stdcpu;
	printf("Context: %p\n", context);
	if(!context)
	{
		std::cerr << "[ERROR] Failed to find Epiphany platform." << std::endl;
		return 2;
	}

	unsigned int devnum = 0;

	clopen(context, NULL, CLLD_NOW);
	
	cl_kernel kernel = clsym(context, NULL, "random_test", 0);
	if (!kernel)
	{
		std::cerr << "error: kernel = " << kernel << std::endl;
		return 1;
	}
	printf("kernel: %p\n", kernel);

	// 10 000 000
	unsigned int work_size = 1000000;
	unsigned int workers = 16;

        if(work_size % workers != 0)
	{
                work_size += workers - (work_size % workers);
	}
        unsigned int paths_per_worker = work_size/workers;

	printf("Work size: %u\n", work_size);
	printf("Workers:   %u\n", workers);
	printf("Work p.W.: %u\n", paths_per_worker);

	cl_uint2 *seeds = (cl_uint2*)clmalloc(context, workers*sizeof(cl_uint2), 0);
	for(unsigned int i = 0; i < workers; i++)
	{
		seeds[i].x = rand();
		seeds[i].y = rand();
	}
	printf("Seeds:\n");
	for(unsigned int i = 0; i < workers; i++)
	{
		printf("%u: %u/%u\n", i, seeds[i].x, seeds[i].y);
	}

	clmsync(context, devnum, seeds, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	
	/* allocate OpenCL device-sharable memory */
	cl_float2* results = (cl_float2*)clmalloc(context, workers*sizeof(cl_float2), 0);
	
	for(unsigned int i = 0; i < workers; i++)
	{
		results[i].x = 23.0f;
		results[i].y = 42.0f;
	}
	printf("Setup canarries\n");

	clmsync(context, devnum, results, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	printf("Synced results to device\n");

	/* define the computational domain and workgroup size */
	clndrange_t index_range = clndrange_init1d(0, workers, workers);

	/* non-blocking fork of the OpenCL kernel to execute on the GPU */
	printf("Forking kernel, standby...\n");
	clforka(context, devnum, kernel, &index_range, CL_EVENT_NOWAIT, paths_per_worker, seeds, results);
	printf("We have fork-off!\n");

	/* non-blocking sync vector c to host memory (copy back to host) */
	clmsync(context, devnum,  results, CL_MEM_HOST|CL_EVENT_NOWAIT);
	printf("Synching results back\n");

	/* force execution of operations in command queue (non-blocking call) */
	clflush(context, devnum, 0);
	printf("Queue flushed\n");

	/* block on completion of operations in command queue */
	clwait(context, devnum, CL_ALL_EVENT);
	printf("OpenCL done!\n");

	for(int i = 0; i < workers; i++)
	{
		printf("Worker %d:\n\tMean:     %10.5f\n\tVariance: %10.5f\n", i, results[i].x, results[i].y);
	}

	float population_mean, population_variance;
	population_statistics_from_subsample_statistics(workers, paths_per_worker, results, &population_mean, &population_variance);

	printf("Population statistics:\n");
	printf("\tPopulation size: %10u\n", work_size);
	printf("\tMean:            %10.5f\n", population_mean);
	printf("\tVariance:        %10.5f\n", population_variance);

	clfree(seeds);
	clfree(results);

	return 0;
}
