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
	bool use_geometric_control_variate = true;

	unsigned int total_number_of_paths = 100000;
	unsigned int workers = 16;
	// END

	// make sure problem size (total number of paths) is divisible by number of workers
	if(total_number_of_paths % workers != 0)
	{
		total_number_of_paths += workers - (total_number_of_paths % workers);
	}
	unsigned int paths_per_worker = total_number_of_paths/workers;

	std::cout << "Inputs:" << std::endl;
	printf("\tStart price:    %10.2f HKD\n", start_price);
	printf("\tStrike price:   %10.2f HKD\n", strike_price);
	printf("\tMaturity:       %10.3f years\n", maturity);
	printf("\tVolatility:     %10.5f %%\n", volatility*100);
	printf("\tRisk-free rate: %10.5f %%\n", risk_free_rate*100);
	printf("\tSteps:          %10d\n", averaging_steps);
	printf("\tPaths:          %10d\n", total_number_of_paths);
	printf("\tWorkers:        %10d\n", workers);
	printf("\tP.p.W.:         %10d\n", paths_per_worker);
	if(use_geometric_control_variate)
	{
		printf("\tCV:              Geometric\n");
	}
	else
	{
		printf("\tCV:                   None\n");
	}

	printf("\n\nRunning...\n");

	srand(time(NULL));

	stdcl_init();

	/* use default contexts, if no GPU use CPU */
	CLCONTEXT* context = (stdgpu)? stdgpu : stdcpu;

	unsigned int devnum = 0;

	clopen(context, NULL, CLLD_NOW);
	
	cl_kernel kernel_no_cv = clsym(context, NULL, "arithmetic_asian_no_cv", 0);
	cl_kernel kernel_geometric_cv = clsym(context, NULL, "arithmetic_asian_geometric_cv", 0);
	if (!kernel_no_cv || !kernel_geometric_cv)
	{
		std::cerr << "error: kernel_no_cv = " << kernel_no_cv << ", kernel_geometric_cv = " << kernel_geometric_cv << std::endl;
		return 1;
	}

	cl_uint2 *seeds = generate_seeds(context, workers);
	clmsync(context, devnum, seeds, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	
	/* allocate OpenCL device-sharable memory */
	cl_float2 *results;
	cl_float2 *arithmetic_results;
	cl_float2 *geometric_results;
	cl_float *arithmetic_geometric_means;

	if(use_geometric_control_variate)
	{
		arithmetic_results = (cl_float2*)clmalloc(context, workers*sizeof(cl_float2), 0);
		geometric_results = (cl_float2*)clmalloc(context, workers*sizeof(cl_float2), 0);       
		arithmetic_geometric_means = (cl_float*)clmalloc(context, workers*sizeof(cl_float), 0);       
		for(int i = 0; i < workers; i++)
		{
			// canaries, as a sanity test
			arithmetic_results[i].x = 23.0f;
			arithmetic_results[i].y = 42.0f;

			geometric_results[i].x = 23.42f;
			geometric_results[i].y = 42.23f;

			arithmetic_geometric_means[i] = 65.65f;
		}
		clmsync(context, devnum, arithmetic_results, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
		clmsync(context, devnum, geometric_results, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
		clmsync(context, devnum, arithmetic_geometric_means, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	}
	else
	{
		results = (cl_float2*)clmalloc(context, workers*sizeof(cl_float2), 0);       
		for(int i = 0; i < workers; i++)
		{
			// canaries, as a sanity test
			results[i].x = 23.0f;
			results[i].y = 42.0f;
		}
		clmsync(context, devnum, results, CL_MEM_DEVICE|CL_EVENT_NOWAIT); // sanity test
	}

	/* define the computational domain and workgroup size */
	clndrange_t index_range = clndrange_init1d(0, workers, 1);

	/* non-blocking fork of the OpenCL kernel to execute on the GPU */
	if(use_geometric_control_variate)
	{
		clforka(context, devnum, kernel_geometric_cv, &index_range, CL_EVENT_NOWAIT, direction, start_price, strike_price, maturity, volatility, risk_free_rate, averaging_steps, total_number_of_paths, 0, seeds, arithmetic_results, geometric_results, arithmetic_geometric_means);
	}
	else
	{	
		clforka(context, devnum, kernel_no_cv, &index_range, CL_EVENT_NOWAIT, direction, start_price, strike_price, maturity, volatility, risk_free_rate, averaging_steps, total_number_of_paths, seeds, results);
	}

	/* non-blocking sync vector c to host memory (copy back to host) */
	if(use_geometric_control_variate)
	{
		clmsync(context, devnum, arithmetic_results, CL_MEM_HOST|CL_EVENT_NOWAIT);
		clmsync(context, devnum, geometric_results, CL_MEM_HOST|CL_EVENT_NOWAIT);
		clmsync(context, devnum, arithmetic_geometric_means, CL_MEM_HOST|CL_EVENT_NOWAIT);
	}
	else
	{
		clmsync(context, devnum,  results, CL_MEM_HOST|CL_EVENT_NOWAIT);
	}

	/* force execution of operations in command queue (non-blocking call) */
	clflush(context, devnum, 0);

	/* block on completion of operations in command queue */
	clwait(context, devnum, CL_ALL_EVENT);

	for(int i = 0; i < workers; i++)
	{
		printf("Worker %d:\n", i);
		if(use_geometric_control_variate)
		{
			// printf("\tE(X):     %10.5f\n", arithmetic_results[i].x);
			// printf("\tVar(X):   %10.5f\n", arithmetic_results[i].y);
			// printf("\tE(Y):     %10.5f\n", geometric_results[i].x);
			// printf("\tVar(Y):   %10.5f\n", geometric_results[i].y);
			// printf("\tE(XY):    %10.5f\n", arithmetic_geometric_means[i]);

			printf("\tE(X):     %f\n", arithmetic_results[i].x);
			printf("\tVar(X):   %f\n", arithmetic_results[i].y);
			printf("\tE(Y):     %f\n", geometric_results[i].x);
			printf("\tVar(Y):   %f\n", geometric_results[i].y);
			printf("\tE(XY):    %f\n", arithmetic_geometric_means[i]);
		}
		else
		{
			printf("\tE(X):     %10.5f\n", results[i].x);
			printf("\tVar(X):   %10.5f\n", results[i].y);
		}
	}

	if(use_geometric_control_variate)
	{
		// On host:
                // calculate total mean of arith path price -> E(X)
                // calculate total mean of geom path price -> E(Y)
                // calculate total mean of (arith * geom) path price -> E(XY)
                // calculate total variance of arith path price -> Var(X)
                // calculate total variance of geom path price -> Var(Y)
                // calculate total covariance of X,Y as Cov(X,Y) = E(XY) - E(X)*E(Y)
                // set E(Z) = E(X)
                // calculate theta as \theta = Cov(X,Y)/Var(Y)
                // calculate total variance of Z as Var(Z) = Var(X) - 2\theta*Cov(X,Y) + \theta^2*Var(Y)
                // result is E(Z),Var(Z)

		float arithmetic_variances_mean = 0.0;
		float geometric_variances_mean = 0.0;
		for(int i = 0; i < workers; i++)
		{
			arithmetic_variances_mean += arithmetic_results[i].x;
			geometric_variances_mean += geometric_results[i].x;
		}
		arithmetic_variances_mean /= workers;
		geometric_variances_mean /= workers;
	
		float arithmetic_variances_variance = 0.0;
		float geometric_variances_variance = 0.0;
		for(int i = 0; i < workers; i++)
		{
			arithmetic_variances_variance += (arithmetic_results[i].x - arithmetic_variances_mean)*(arithmetic_results[i].x - arithmetic_variances_mean);
			geometric_variances_variance += (geometric_results[i].x - geometric_variances_mean)*(geometric_results[i].x - geometric_variances_mean);
		}
		arithmetic_variances_variance /= workers;
		geometric_variances_variance /= workers;

		float total_arithmetic_mean = 0.0f;
		float sum_of_arithmetic_variances = 0.0f;

		float total_geometric_mean = 0.0f;
		float sum_of_geometric_variances = 0.0f;

		float total_arithmetic_geometric_mean = 0.0f;

		for(int i = 0; i < workers; i++)
		{
			total_arithmetic_mean += arithmetic_results[i].x * (1.0f/(float)workers);
			total_geometric_mean += geometric_results[i].x * (1.0f/(float)workers);
			total_arithmetic_geometric_mean += arithmetic_geometric_means[i] * (1.0f/(float)workers);

			sum_of_arithmetic_variances += arithmetic_results[i].y;
			sum_of_geometric_variances += geometric_results[i].y;
		}
		float total_arithmetic_variance = ((float)paths_per_worker - 1.0f)/((float)total_number_of_paths - 1.0f) * (sum_of_arithmetic_variances + (((float)paths_per_worker * ((float)workers - 1.0f))/((float)paths_per_worker-1.0f) * arithmetic_variances_variance));
		float total_geometric_variance = ((float)paths_per_worker - 1.0f)/((float)total_number_of_paths - 1.0f) * (sum_of_geometric_variances + (((float)paths_per_worker * ((float)workers - 1.0f))/((float)paths_per_worker-1.0f) * geometric_variances_variance));

                // calculate total covariance of X,Y as Cov(X,Y) = E(XY) - E(X)*E(Y)
		float total_covariance = total_arithmetic_geometric_mean - (total_arithmetic_mean * total_geometric_mean);

                // calculate theta as \theta = Cov(X,Y)/Var(Y)
		float theta = total_covariance/total_geometric_variance;

                // set E(Z) = E(X)
		float total_mean = total_arithmetic_mean;

                // calculate total variance of Z as Var(Z) = Var(X) - 2\theta*Cov(X,Y) + \theta^2*Var(Y)
		float total_variance = total_arithmetic_variance - 2*theta*total_covariance + theta*theta*total_geometric_variance;

		printf("\nTotal population statistics:\n");

		printf("\n");
		printf("\tE(X):      %10.5f\n", total_arithmetic_mean);
		printf("\tVar(X):    %10.5f\n", total_arithmetic_variance);
		printf("\tStdDev(X): %10.5f\n", sqrt(total_arithmetic_variance));

		printf("\n");
		printf("\tE(Y):      %10.5f\n", total_geometric_mean);
		printf("\tVar(Y):    %10.5f\n", total_geometric_variance);
		printf("\tStdDev(Y): %10.5f\n", sqrt(total_geometric_variance));

		printf("\n");
		printf("\tE(XY):     %10.5f\n", total_arithmetic_geometric_mean);
		printf("\tCov(X,Y):  %10.5f\n", total_covariance);
		printf("\tTheta:     %10.5f\n", theta);

		printf("\n");
		printf("\tMean:     %10.5f\n", total_mean);
		printf("\tVariance: %10.5f\n", total_variance);
		printf("\tStdDev:   %10.5f\n", sqrt(total_variance));


		float confidence_interval_lower = total_mean - 1.96*(sqrt(total_variance)/sqrt(total_number_of_paths));
		float confidence_interval_upper = total_mean + 1.96*(sqrt(total_variance)/sqrt(total_number_of_paths));
		printf("\tCI:     [ %10.7f,\n\t          %10.7f ]\n", confidence_interval_lower, confidence_interval_upper);
		printf("\tCI size:  %10.7f\n", confidence_interval_upper-confidence_interval_lower);

		clfree(arithmetic_results);
		clfree(geometric_results);
		clfree(arithmetic_geometric_means);
	}
	else
	{
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
	
		float total_mean = 0.0f;
		float sum_of_variances = 0.0f;

		for(int i = 0; i < workers; i++)
		{
			total_mean += results[i].x * (1.0f/(float)workers);

			sum_of_variances += results[i].y;
		}
		float total_variance = ((float)paths_per_worker - 1.0f)/((float)total_number_of_paths - 1.0f) * (sum_of_variances + (((float)paths_per_worker * ((float)workers - 1.0f))/((float)paths_per_worker-1.0f) * variances_variance));

		printf("\nTotal population statistics:\n");
		printf("\tMean:     %10.5f\n", total_mean);
		printf("\tVariance: %10.5f\n", total_variance);
		printf("\tStdDev:   %10.5f\n", sqrt(total_variance));

		float confidence_interval_lower = total_mean - 1.96*(sqrt(total_variance)/sqrt(total_number_of_paths));
		float confidence_interval_upper = total_mean + 1.96*(sqrt(total_variance)/sqrt(total_number_of_paths));
		printf("\tCI:     [ %10.7f,\n\t          %10.7f ]\n", confidence_interval_lower, confidence_interval_upper);
		printf("\tCI size:  %10.7f\n", confidence_interval_upper-confidence_interval_lower);

		clfree(results);
	}

	clfree(seeds);

	return 0;
}
