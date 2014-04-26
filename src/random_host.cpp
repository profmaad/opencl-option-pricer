/* (c) 2014 Maximilian Gerhard Wolter */

# include <iostream>

# include <cstdio>
# include <cstdlib>
# include <ctime>
# include <cmath>

# include <stdcl.h>

# include "statistics_opencl.h"

float arithmetic_mean(unsigned int n, float *values)
{
	float mean = 0.0;

	float factor = 1.0f/(float)n;

	for(int i = 0; i < n; i++)
	{
		mean += values[i] * factor;
	}

	return mean;
}

void initialize_running_variance(unsigned int *iteration, float *mean, float *m2)
{
	*iteration = 0;
	*mean = 0;
	*m2 = 0;
}
void update_running_variance(unsigned int *iteration, float *mean, float *m2, float sample)
{
	*iteration += 1;

	float new_mean = *mean + (sample - *mean) / *iteration;

	float new_m2 = *m2 + (sample - *mean) * (sample - new_mean);

	*mean = new_mean;
	*m2 = new_m2;
}
float finalize_running_variance(unsigned int *iteration, float *mean, float *m2)
{
	return *m2 / ((*iteration)-1);
}

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
	
	cl_kernel kernel = clsym(context, NULL, "random_test_host", 0);
	if (!kernel)
	{
		std::cerr << "error: kernel = " << kernel << std::endl;
		return 1;
	}
	printf("kernel: %p\n", kernel);

	// 10 000 000
	unsigned int work_size = 10000000;
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
	cl_float* results = (cl_float*)clmalloc(context, work_size*sizeof(cl_float), 0);
	
	for(unsigned int i = 0; i < work_size; i++)
	{
		results[i] = 0.0f;
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

	float population_mean, population_variance;

	population_mean = arithmetic_mean(work_size, results);
	population_variance = 0.0f;

	for(unsigned int i = 0; i < work_size; i++)
	{
		population_variance += pow(results[i] - population_mean, 2)/(work_size-1);
	}

	printf("Population statistics:\n");
	printf("\tPopulation size: %10u\n", work_size);
	printf("\tMean:            %10.5f\n", population_mean);
	printf("\tVariance:        %10.5f\n", sqrt(population_variance));
	printf("\tStdDev:          %10.5f\n", population_variance);

	clfree(seeds);
	clfree(results);

	return 0;
}
