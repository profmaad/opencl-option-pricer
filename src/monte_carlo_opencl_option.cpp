/* (c) 2014 Maximilian Gerhard Wolter */

# include <cassert>

# include <stdcl.h>

# include "statistics.h"
# include "statistics_opencl.h"
# include "json_helper.hpp"

# include "monte_carlo_opencl_option.hpp"

MonteCarloOpenCLOption::MonteCarloOpenCLOption(JSONHelper &parameters) : OpenCLOption(parameters),
									 arithmetic_results(NULL),
									 geometric_results(NULL),
									 arithmetic_geometric_means(NULL)
{
	control_variate = parameters.get_control_variate();
	number_of_paths = parameters.get_uint("samples");

	opencl_configuration_changed();
}

void MonteCarloOpenCLOption::opencl_configuration_changed()
{
	if(number_of_paths % number_of_workers != 0)
	{
		number_of_paths += number_of_workers - (number_of_paths % number_of_workers);
	}
	paths_per_worker = number_of_paths/number_of_workers;

	assert(number_of_paths > 0);
	assert(paths_per_worker > 0);
}

bool MonteCarloOpenCLOption::use_control_variate()
{
	return (control_variate == Geometric || control_variate == Geometric_AdjustedStrike);
}
bool MonteCarloOpenCLOption::use_adjusted_strike()
{
	return control_variate == Geometric_AdjustedStrike;
}

void MonteCarloOpenCLOption::setup_outputs()
{
	arithmetic_results = (cl_float2*)clmalloc(context, number_of_workers*sizeof(cl_float2), 0);
	assert(arithmetic_results != NULL);
	clmsync(context, device_number, arithmetic_results, CL_MEM_DEVICE|CL_EVENT_NOWAIT);

	if(use_control_variate())
	{
		geometric_results = (cl_float2*)clmalloc(context, number_of_workers*sizeof(cl_float2), 0);
		arithmetic_geometric_means = (cl_float*)clmalloc(context, number_of_workers*sizeof(cl_float), 0);
	
		assert(geometric_results != NULL);
		assert(arithmetic_geometric_means != NULL);

		clmsync(context, device_number, geometric_results, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
		clmsync(context, device_number, arithmetic_geometric_means, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	}
}

void MonteCarloOpenCLOption::retrieve_results()
{
	clmsync(context, device_number, arithmetic_results, CL_MEM_HOST|CL_EVENT_NOWAIT);

	if(use_control_variate())
	{
		assert(geometric_results != NULL);
		assert(arithmetic_geometric_means != NULL);

		clmsync(context, device_number, geometric_results, CL_MEM_HOST|CL_EVENT_NOWAIT);
		clmsync(context, device_number, arithmetic_geometric_means, CL_MEM_HOST|CL_EVENT_NOWAIT);
	}
}

void MonteCarloOpenCLOption::finalize_results(float *population_mean, float *confidence_interval_lower, float *confidence_interval_upper)
{
	float population_variance;

	float arithmetic_mean, arithmetic_variance;
	population_statistics_from_subsample_statistics(number_of_workers, paths_per_worker, arithmetic_results, &arithmetic_mean, &arithmetic_variance);

	if(use_control_variate())
	{
		float geometric_mean, geometric_variance;
		float arithmetic_geometric_mean;

		population_statistics_from_subsample_statistics(number_of_workers, paths_per_worker, geometric_results, &geometric_mean, &geometric_variance);
		arithmetic_geometric_mean = population_mean_from_subsample_mean(number_of_workers, arithmetic_geometric_means);

                // calculate covariance of X,Y as Cov(X,Y) = E(XY) - E(X)*E(Y)
		float covariance = arithmetic_geometric_mean - (arithmetic_mean * geometric_mean);

                // calculate theta as \theta = Cov(X,Y)/Var(Y)
		float theta = covariance/geometric_variance;

                // set E(Z) = E(X)
		*population_mean = arithmetic_mean;

                // calculate total variance of Z as Var(Z) = Var(X) - 2\theta*Cov(X,Y) + \theta^2*Var(Y)
		population_variance = arithmetic_variance - 2*theta*covariance + theta*theta*geometric_variance;

		if(population_variance < 0)
		{
			fprintf(stderr, "[WARNING]: population variance is negative, assuming variance of zero.\n");
			population_variance = 0;
		}
	}
	else
	{
		*population_mean = arithmetic_mean;
		population_variance = arithmetic_variance;
	}

	confidence_interval_95percent(number_of_paths, *population_mean, population_variance, confidence_interval_lower, confidence_interval_upper);
}

void MonteCarloOpenCLOption::cleanup()
{
	if(arithmetic_results) { clfree(arithmetic_results); }
	if(geometric_results) { clfree(geometric_results); }
	if(arithmetic_geometric_means) { clfree(arithmetic_geometric_means); }
	
	arithmetic_results = NULL;
	geometric_results = NULL;
	arithmetic_geometric_means = NULL;
}
