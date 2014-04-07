/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <iostream>

# include <cstdio>
# include <cstdlib>
# include <ctime>
# include <cmath>

# include <stdcl.h>

cl_uint2* generate_seeds(CLCONTEXT* context, unsigned int workers)
{
	cl_uint2 *seeds = (cl_uint2*)clmalloc(context, workers*sizeof(cl_uint2), 0);
	for(int i = 0; i < workers; i++)
	{
		seeds[i].x = rand();
		seeds[i].y = rand();
	}

	return seeds;
}

# define CALL 0
# define PUT 1

int main(int argc, char **argv)
{
	// problem definition
	unsigned int direction = PUT;
	float start_price = 100.0;
	float strike_price = 100.0;
	float maturity = 3.0;
	float volatility = 0.3;
	float risk_free_rate = 0.05;
	unsigned int averaging_steps = 50;

	unsigned int total_number_of_paths = 1000000;
	unsigned int workers = 16;
	// END

	// make sure problem size (total number of paths) is divisible by number of workers
	total_number_of_paths += workers - (total_number_of_paths % workers);       

	std::cout << "Inputs:" << std::endl;
	printf("\tStart price:    %10.2f HKD\n", start_price);
	printf("\tStrike price:   %10.2f HKD\n", strike_price);
	printf("\tMaturity:       %10.3f years\n", maturity);
	printf("\tVolatility:     %10.5f %%\n", volatility*100);
	printf("\tRisk-free rate: %10.5f %%\n", risk_free_rate*100);
	printf("\tSteps:          %10d\n", averaging_steps);
	printf("\tPaths:          %10d\n", total_number_of_paths);
	printf("\tWorkers:        %10d\n", workers);
	printf("\tP.p.W.:         %10d\n", total_number_of_paths/workers);

	printf("\n\nRunning...\n");

	srand(time(NULL));

	stdcl_init();

	/* use default contexts, if no GPU use CPU */
	CLCONTEXT* context = (stdgpu)? stdgpu : stdcpu;

	unsigned int devnum = 0;

	clopen(context, NULL, CLLD_NOW);
	
	cl_kernel kernel = clsym(context, NULL, "arithmetic_asian_no_cv", 0);
	if (!kernel)
	{
		std::cerr << "error: kernel = " << kernel << std::endl;
		return 1;
	}

	cl_uint2 *seeds = generate_seeds(context, workers);
	clmsync(context, devnum, seeds, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	
	/* allocate OpenCL device-sharable memory */
	cl_float2* results = (cl_float2*)clmalloc(context, workers*sizeof(cl_float2), 0);       
	for(int i = 0; i < workers; i++)
	{
		// canaries, as a sanity test
		results[i].x = 23.0f;
		results[i].y = 42.0f;
	}
	clmsync(context, devnum, results, CL_MEM_DEVICE|CL_EVENT_NOWAIT); // sanity test

	/* define the computational domain and workgroup size */
	clndrange_t index_range = clndrange_init1d(0, workers, 1);

	/* non-blocking fork of the OpenCL kernel to execute on the GPU */
	clforka(context, devnum, kernel, &index_range, CL_EVENT_NOWAIT, direction, start_price, strike_price, maturity, volatility, risk_free_rate, averaging_steps, total_number_of_paths, seeds, results);

	/* non-blocking sync vector c to host memory (copy back to host) */
	clmsync(context, devnum,  results, CL_MEM_HOST|CL_EVENT_NOWAIT);

	/* force execution of operations in command queue (non-blocking call) */
	clflush(context, devnum, 0);

	/* block on completion of operations in command queue */
	clwait(context, devnum, CL_ALL_EVENT);

	for(int i = 0; i < workers; i++)
	{
		printf("Worker %d:\n\tMean:     %10.5f\n\tVariance: %10.5f\n\tstddev:   %10.5f\n", i, results[i].x, results[i].y, sqrt(results[i].y));
	}

	// calculate statistics for total sample population
	float variances_mean = 0.0;
	for(int i = 0; i < workers; i++)
	{
		variances_mean += results[i].x;
	}
	variances_mean /= workers;
	
	float variances_variance = 0.0;
	for(int i = 0; i < workers; i++)
	{
		variances_variance += (results[i].x - variances_mean)*(results[i].x - variances_mean);
	}
	variances_variance /= workers;
	
	unsigned int paths_per_worker = total_number_of_paths/workers;
	float total_mean = 0.0f;
	float sum_of_variances = 0.0f;

	for(int i = 0; i < workers; i++)
	{
		total_mean += results[i].x * (1.0f/(float)workers);

		sum_of_variances += results[i].y;
	}
	float total_variance = ((float)paths_per_worker - 1.0f)/((float)total_number_of_paths - 1.0f) * (sum_of_variances + (((float)paths_per_worker * ((float)workers - 1.0f))/((float)paths_per_worker-1.0f) * variances_variance));

	printf("\nTotal population statistics:\n\tMean:     %10.5f\n\tVariance: %10.5f\n\tstddev:   %10.5f\n", total_mean, total_variance, sqrt(total_variance));

	float confidence_interval_lower = total_mean - 1.96*(sqrt(total_variance)/sqrt(total_number_of_paths));
	float confidence_interval_upper = total_mean + 1.96*(sqrt(total_variance)/sqrt(total_number_of_paths));
	printf("\tCI:     [ %10.7f,\n\t          %10.7f ]\n", confidence_interval_lower, confidence_interval_upper);
	printf("\tCI size:  %10.7f\n", confidence_interval_upper-confidence_interval_lower);

	clfree(seeds);
	clfree(results);

	return 0;
}
