/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <iostream>
# include <string>

# include <cstdio>
# include <cstdlib>
# include <ctime>
# include <cmath>

# include <stdcl.h>

# include "matrix.h"
# include "statistics.h"
# include "statistics_opencl.h"

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

enum OptionType
{
	European = 0,
	Asian_Geometric,
	Asian_Arithmetic,
	Basket_Geometric,
	Basket_Arithmetic,

	OptionType_SIZE
};
enum OptionDirection
{
	Call = 0,
	Put,

	OptionDirection_SIZE
};
enum ControlVariate
{
	None = 0,
	Geometric,
	Geometric_AdjustedStrike,

	ControlVariate_SIZE
};

static const char *KERNEL_SYMBOLS[OptionType_SIZE][ControlVariate_SIZE] = { 
	{"european", "european", "european"},
	{"geometric_asian", "geometric_asian", "geometric_asian"},
	{"arithmetic_asian_no_cv", "arithmetic_asian_geometric_cv", "arithmetic_asian_geometric_cv"},
	{"geometric_basket", "geometric_basket", "geometric_basket"},
	{"arithmetic_basket_no_cv", "arithmetic_basket_geometric_cv", "arithmetic_basket_geometric_cv"}
};

const char* get_kernel_sym(OptionType type, ControlVariate control_variate)
{
	if(type >= OptionType_SIZE || control_variate >= ControlVariate_SIZE) { return NULL; }
	else { return KERNEL_SYMBOLS[type][control_variate]; }
}

# define NUMBER_OF_ASSETS 3

int main(int argc, char **argv)
{
// problem definition
	OptionType type = Basket_Arithmetic;
	OptionDirection direction = Call;
	ControlVariate control_variate = Geometric_AdjustedStrike;
	unsigned int number_of_assets = NUMBER_OF_ASSETS;
	float start_prices[NUMBER_OF_ASSETS] = {100.0, 90.0, 110.0};
	float strike_price = 100.0;
	float maturity = 3.0;
	float volatilities[NUMBER_OF_ASSETS] = {0.3, 0.15, 0.10};
	float risk_free_rate = 0.05;
	//float correlations[NUMBER_OF_ASSETS*NUMBER_OF_ASSETS] = {1.0, 0.8, 0.8, 1.0};
	float correlations[NUMBER_OF_ASSETS*NUMBER_OF_ASSETS] = {1.0, 0.8, 0.5, 0.8, 1.0, 0.3, 0.5, 0.3, 1.0};
	unsigned int averaging_steps = 50;

	unsigned int total_number_of_paths = 1000000;
	unsigned int workers = 16;
	// END

	// make sure problem size (total number of paths) is divisible by number of workers
	if(total_number_of_paths % workers != 0)
	{
		total_number_of_paths += workers - (total_number_of_paths % workers);
	}
	unsigned int paths_per_worker = total_number_of_paths/workers;
	
	bool use_geometric_control_variate = false;
	unsigned int use_adjusted_strike = 0;
	if(type == Asian_Arithmetic || type == Basket_Arithmetic)
	{
		switch(control_variate)
		{
		case None:
			use_geometric_control_variate = false;
			use_adjusted_strike = 0;
			break;
		case Geometric:
			use_geometric_control_variate = true;
			use_adjusted_strike = 0;
			break;
		case Geometric_AdjustedStrike:
			use_geometric_control_variate = true;
			use_adjusted_strike = 1;
			break;
		}
	}

	// calculate cholesky decomposition of asset correlation matrix (needed for generation of correlated random numbers)
	float correlations_cholesky[number_of_assets*number_of_assets];
	cholesky_decomposition(number_of_assets, correlations, correlations_cholesky);

	std::cout << "Inputs:" << std::endl;
	printf("\tOption Type:    ");
	switch(type)
	{
	case European:
		printf("                  European");
		break;
	case Asian_Geometric:
		printf("           Geometric Asian");
		break;
	case Asian_Arithmetic:
		printf("          Arithmetic Asian");
		break;
	case Basket_Geometric:
		printf("          Geometric Basket");
		break;
	case Basket_Arithmetic:
		printf("         Arithmetic Basket");
		break;
	default:
		printf("          Unknown");
	}
	printf("\n");
	printf("\tDirection:      ");
	switch(direction)
	{
	case Call:
		printf("                      Call");
		break;
	case Put:
		printf("                       Put");
		break;
	default:
		printf("                   Unknown");
	}
	printf("\n");
	switch(control_variate)
	{
	case None:
		printf("\tCV:                                   None\n");
		break;
	case Geometric:
		printf("\tCV:                              Geometric\n");
		break;
	case Geometric_AdjustedStrike:
		printf("\tCV:             Geometric, adjusted strike\n");
		break;
	default:
		printf("\tCV:                                Unknown\n");
	}

	for(int asset = 0; asset < number_of_assets; asset++)
	{
		printf("\tStart price(%d): %10.2f HKD\n", asset, start_prices[asset]);
		printf("\tVolatility (%d): %10.5f %%\n", asset, volatilities[asset]*100);
	}
	printf("\tStrike price:   %10.2f HKD\n", strike_price);
	printf("\tMaturity:       %10.3f years\n", maturity);
	printf("\tRisk-free rate: %10.5f %%\n", risk_free_rate*100);
	printf("\tSteps:          %10d\n", averaging_steps);
	printf("\tCorrelations:\n");
	for(int row = 0; row < number_of_assets; row++)
	{
		printf("\t                ");
		for(int column = 0; column < number_of_assets; column++)
		{
			printf("%5.3f ", correlations[row*number_of_assets + column]);
		}
		printf("\n");
	}
	printf("\tCholesky Dec.:\n");
	for(int row = 0; row < number_of_assets; row++)
	{
		printf("\t                ");
		for(int column = 0; column < number_of_assets; column++)
		{
			printf("%5.3f ", correlations_cholesky[row*number_of_assets + column]);
		}
		printf("\n");
	}
	printf("\tPaths:          %10d\n", total_number_of_paths);
	printf("\tWorkers:        %10d\n", workers);
	printf("\tP.p.W.:         %10d\n", paths_per_worker);

	srand(time(NULL));

	stdcl_init();

	/* use default contexts, if no GPU use CPU */
	CLCONTEXT* context = (stdgpu)? stdgpu : stdcpu;

	unsigned int devnum = 0;

	clopen(context, NULL, CLLD_NOW);

	const char *kernel_sym = get_kernel_sym(type, control_variate);
	if(!kernel_sym)
	{
		std::cerr << "error: kernel_sym == NULL;" << std::endl;
		return 1;
	}
	cl_kernel kernel = clsym(context, NULL, kernel_sym, 0);
	if (!kernel)
	{
		std::cerr << "error: kernel = " << kernel << " for kernel_sym = " << kernel_sym << std::endl;
		return 1;
	}
	printf("\n\nRunning kernel %s...\n", kernel_sym);

	cl_uint2 *seeds = generate_seeds(context, workers);
	clmsync(context, devnum, seeds, CL_MEM_DEVICE|CL_EVENT_NOWAIT);

	// start_prices, asset_volatilities, correlations, correlations_cholesky, seeds
	cl_float *cl_start_prices = (cl_float*)clmalloc(context, number_of_assets*sizeof(cl_float), 0);
	cl_float *cl_volatilities = (cl_float*)clmalloc(context, number_of_assets*sizeof(cl_float), 0);
	cl_float *cl_correlations = (cl_float*)clmalloc(context, number_of_assets*number_of_assets*sizeof(cl_float), 0);
	cl_float *cl_correlations_cholesky = (cl_float*)clmalloc(context, number_of_assets*number_of_assets*sizeof(cl_float), 0);

	for(int asset = 0; asset < number_of_assets; asset++)
	{
		cl_start_prices[asset] = start_prices[asset];
		cl_volatilities[asset] = volatilities[asset];
		
		for(int column = 0; column < number_of_assets; column++)
		{
			cl_correlations[asset * number_of_assets + column] = correlations[asset * number_of_assets + column];
			cl_correlations_cholesky[asset * number_of_assets + column] = correlations_cholesky[asset * number_of_assets + column];
		}
	}
	clmsync(context, devnum, cl_start_prices, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, devnum, cl_volatilities, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, devnum, cl_correlations, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, devnum, cl_correlations_cholesky, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	
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
		switch(type)
		{
		case Asian_Arithmetic:
			clforka(context, devnum, kernel, &index_range, CL_EVENT_NOWAIT, direction, start_prices[0], strike_price, maturity, volatilities[0], risk_free_rate, averaging_steps, total_number_of_paths, use_adjusted_strike, seeds, arithmetic_results, geometric_results, arithmetic_geometric_means);
			break;
		case Basket_Arithmetic:
			clforka(context, devnum, kernel, &index_range, CL_EVENT_NOWAIT, direction, number_of_assets, cl_start_prices, strike_price, maturity, cl_volatilities, risk_free_rate, cl_correlations, cl_correlations_cholesky, total_number_of_paths, use_adjusted_strike, seeds, arithmetic_results, geometric_results, arithmetic_geometric_means);
			break;
		}
	}
	else
	{	
		switch(type)
		{
		case Asian_Arithmetic:
			clforka(context, devnum, kernel, &index_range, CL_EVENT_NOWAIT, direction, start_prices[0], strike_price, maturity, volatilities[0], risk_free_rate, averaging_steps, total_number_of_paths, seeds, results);
			break;
		case Basket_Arithmetic:
			clforka(context, devnum, kernel, &index_range, CL_EVENT_NOWAIT, direction, number_of_assets, cl_start_prices, strike_price, maturity, cl_volatilities, risk_free_rate, cl_correlations, cl_correlations_cholesky, total_number_of_paths, seeds, results);
			break;
		}
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

		// calculate total variance from subsample variance:
		// E(X) = (SUM E(X_i))/n
		// Var(E(X)) = (SUM (E(X_i) - E(X))^2)/n
		// Var(X) = (k-1)/(gk-1) * (SUM Var(X_i) + (k(g-1))/(k-1) * Var(E(X)));
		// Complexity: O(3n)

		// calculate total mean from subsample mean:
		// E(X) = 1/workers * (SUM E(X_i))
		// Complexity: O(n)

		float total_arithmetic_mean, total_arithmetic_variance;
		float total_geometric_mean, total_geometric_variance;
		float total_arithmetic_geometric_mean;

		population_statistics_from_subsample_statistics(workers, paths_per_worker, arithmetic_results, &total_arithmetic_mean, &total_arithmetic_variance);
		population_statistics_from_subsample_statistics(workers, paths_per_worker, geometric_results, &total_geometric_mean, &total_geometric_variance);
		total_arithmetic_geometric_mean = population_mean_from_subsample_mean(workers, arithmetic_geometric_means);

                // calculate total covariance of X,Y as Cov(X,Y) = E(XY) - E(X)*E(Y)
		float total_covariance = total_arithmetic_geometric_mean - (total_arithmetic_mean * total_geometric_mean);

                // calculate theta as \theta = Cov(X,Y)/Var(Y)
		float theta = total_covariance/total_geometric_variance;

                // set E(Z) = E(X)
		float total_mean = total_arithmetic_mean;

                // calculate total variance of Z as Var(Z) = Var(X) - 2\theta*Cov(X,Y) + \theta^2*Var(Y)
		float total_variance = total_arithmetic_variance - 2*theta*total_covariance + theta*theta*total_geometric_variance;

		float confidence_interval_lower, confidence_interval_upper;
		confidence_interval_95percent(total_number_of_paths, total_mean, total_variance, &confidence_interval_lower, &confidence_interval_upper);

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
		printf("\tMean:      %10.5f\n", total_mean);
		printf("\tVariance:  %10.5f\n", total_variance);
		printf("\tStdDev:    %10.5f\n", sqrt(total_variance));

		printf("\tCI:      [ %10.7f,\n\t           %10.7f ]\n", confidence_interval_lower, confidence_interval_upper);
		printf("\tCI size:   %10.7f\n", confidence_interval_upper-confidence_interval_lower);

		clfree(arithmetic_results);
		clfree(geometric_results);
		clfree(arithmetic_geometric_means);
	}
	else
	{
		float total_mean, total_variance;
		population_statistics_from_subsample_statistics(workers, paths_per_worker, results, &total_mean, &total_variance);

		float confidence_interval_lower, confidence_interval_upper;
		confidence_interval_95percent(total_number_of_paths, total_mean, total_variance, &confidence_interval_lower, &confidence_interval_upper);

		printf("\nTotal population statistics:\n");
		printf("\tMean:      %10.5f\n", total_mean);
		printf("\tVariance:  %10.5f\n", total_variance);
		printf("\tStdDev:    %10.5f\n", sqrt(total_variance));

		printf("\tCI:      [ %10.7f,\n\t           %10.7f ]\n", confidence_interval_lower, confidence_interval_upper);
		printf("\tCI size:   %10.7f\n", confidence_interval_upper-confidence_interval_lower);

		clfree(results);
	}

	clfree(seeds);

	return 0;
}
