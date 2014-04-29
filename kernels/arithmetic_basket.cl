/* (c) 2014 Maximilian Gerhard Wolter */

# include <random.h>
# include <statistics.h>

# include "geometric_basket.cl"

# define CALL 0
# define PUT 1

float arithmetic_basket_start_price(unsigned int number_of_assets, __global float *start_prices)
{
	float factor = 1.0f/number_of_assets;

	float start_price = 0.0f;
	for(int asset = 0; asset < number_of_assets; asset++)
	{
		start_price += start_prices[asset]*factor;
	}

	return start_price;
}

float arithmetic_basket_expected_underlying_price_at_maturity(unsigned int number_of_assets, __global float *start_prices, float risk_free_rate, float maturity)
{
	return exp(risk_free_rate * maturity) * arithmetic_basket_start_price(number_of_assets, start_prices);
}

float arithmetic_basket_geometric_cv_adjusted_strike(unsigned int number_of_assets, __global float *start_prices, float strike_price, float maturity, __global float *asset_volatilities, float risk_free_rate, __global float *correlations)
{
	float geometric_expectation = geometric_basket_expected_underlying_price_at_maturity(number_of_assets, start_prices, maturity, asset_volatilities, risk_free_rate, correlations);
	float arithmetic_expectation = arithmetic_basket_expected_underlying_price_at_maturity(number_of_assets, start_prices, risk_free_rate, maturity);

	return strike_price + geometric_expectation - arithmetic_expectation;
}

__kernel void arithmetic_basket_no_cv(unsigned int direction, unsigned int number_of_assets, __global float *start_prices, float strike_price, float maturity, __global float *asset_volatilities, float risk_free_rate, __global float *correlations, __global float *correlations_cholesky, unsigned int total_number_of_paths, __global uint2 *seeds, __global float *uncorrelated_random_global, __global float *random_global, __global float *drifts_global, __global float2 *results)
{
	// get details on worker setup
	unsigned int tid = get_global_id(0);
	unsigned int worker_count = get_global_size(0);

	// calculate work item size
	unsigned int number_of_paths = total_number_of_paths/worker_count;

	// setup PRNG
	uint2 seed = seeds[tid];

	uniform_int_prng_state int_base_state;
	initialize_uniform_int_prng(&int_base_state, seed);
	stdnormal_float_prng_state float_base_state;
	initialize_stdnormal_float_prng(&float_base_state, &int_base_state);
	correlated_stdnormal_float_prng_state prng_state;
	initialize_correlated_stdnormal_float_prng(&prng_state, &float_base_state, number_of_assets, correlations_cholesky, &(uncorrelated_random_global[tid*number_of_assets]));

	// setup running mean and variance calculation
	unsigned int statistics_iteration;
	float running_mean;
	float running_m2;
	initialize_running_variance(&statistics_iteration, &running_mean, &running_m2);
	
	// calculate fixed monte carlo parameters
	__global float *drifts = &(drifts_global[tid*number_of_assets]);
	for(int asset = 0; asset < number_of_assets; asset++)
	{
		drifts[asset] = exp((risk_free_rate - 0.5*asset_volatilities[asset]*asset_volatilities[asset]) * maturity);
	}
	float discounting_factor = exp(-risk_free_rate * maturity);
	__global float *random = &(random_global[tid*number_of_assets]);
	float running_mean_sample_factor = 1.0f / (float)number_of_assets;

	for(int path = 0; path < number_of_paths; path++)
	{
		float path_mean = 0;

		correlated_stdnormal_float_random(&prng_state, random);
		
		for(int asset = 0; asset < number_of_assets; asset++)
		{
			float growth_factor = drifts[asset] * exp(asset_volatilities[asset] * sqrt(maturity) * random[asset]);
			float asset_price = start_prices[asset] * growth_factor;
			path_mean += asset_price*running_mean_sample_factor;
		}

		// calculate payoff - save variables...
		path_mean = (direction == CALL ? max(path_mean - strike_price, 0.0f) : max(strike_price - path_mean, 0.0f));
		// calculate discounted value
		path_mean *= discounting_factor;

		// update sample statistics with new path value
		update_running_variance(&statistics_iteration, &running_mean, &running_m2, path_mean);
	}

	results[tid].x = running_mean;
	results[tid].y = finalize_running_variance(&statistics_iteration, &running_mean, &running_m2);
}

__kernel void arithmetic_basket_geometric_cv(unsigned int direction, unsigned int number_of_assets, __global float *start_prices, float strike_price, float maturity, __global float *asset_volatilities, float risk_free_rate, __global float *correlations, __global float *correlations_cholesky, unsigned int total_number_of_paths, unsigned int adjust_strike, __global uint2 *seeds, __global float *uncorrelated_random_global, __global float *random_global, __global float *drifts_global, __global float2 *arithmetic_results, __global float2 *geometric_results, __global float *arithmetic_geometric_means)
{
	// get details on worker setup
	unsigned int tid = get_global_id(0);
	unsigned int worker_count = get_global_size(0);

	// calculate work item size
	unsigned int number_of_paths = total_number_of_paths/worker_count;

	// setup PRNG
	uint2 seed = seeds[tid];

	uniform_int_prng_state int_base_state;
	initialize_uniform_int_prng(&int_base_state, seed);
	stdnormal_float_prng_state float_base_state;
	initialize_stdnormal_float_prng(&float_base_state, &int_base_state);
	correlated_stdnormal_float_prng_state prng_state;
	initialize_correlated_stdnormal_float_prng(&prng_state, &float_base_state, number_of_assets, correlations_cholesky, &(uncorrelated_random_global[tid*number_of_assets]));

	// setup running mean and variance calculation
	unsigned int arithmetic_iterations;
	float running_arithmetic_mean;
	float running_arithmetic_m2;
	initialize_running_variance(&arithmetic_iterations, &running_arithmetic_mean, &running_arithmetic_m2);

	unsigned int geometric_iterations;
	float running_geometric_mean;
	float running_geometric_m2;
	initialize_running_variance(&geometric_iterations, &running_geometric_mean, &running_geometric_m2);

	unsigned int arithmetic_geometric_iterations;
	float running_arithmetic_geometric_mean;
	float running_arithmetic_geometric_m2;
	initialize_running_variance(&arithmetic_geometric_iterations, &running_arithmetic_geometric_mean, &running_arithmetic_geometric_m2);

	// calculate fixed monte carlo parameters
	__global float *drifts = &(drifts_global[tid*number_of_assets]);
	for(int asset = 0; asset < number_of_assets; asset++)
	{
		drifts[asset] = exp((risk_free_rate - 0.5*asset_volatilities[asset]*asset_volatilities[asset]) * maturity);
	}
	float discounting_factor = exp(-risk_free_rate * maturity);
	__global float *random = &(random_global[tid*number_of_assets]);
	float path_mean_sample_factor = 1.0f / (float)number_of_assets;

	float adjusted_strike_price = (adjust_strike == 0 ? strike_price : arithmetic_basket_geometric_cv_adjusted_strike(number_of_assets, start_prices, strike_price, maturity, asset_volatilities, risk_free_rate, correlations));

	for(int path = 0; path < number_of_paths; path++)
	{
		float path_arithmetic_mean = 0;
		float path_geometric_mean = 0;

		correlated_stdnormal_float_random(&prng_state, random);
		
		for(int asset = 0; asset < number_of_assets; asset++)
		{
			float growth_factor = drifts[asset] * exp(asset_volatilities[asset] * sqrt(maturity) * random[asset]);
			float asset_price = start_prices[asset] * growth_factor;

			path_arithmetic_mean += asset_price*path_mean_sample_factor;
			path_geometric_mean += log(asset_price)*path_mean_sample_factor;
		}

		path_geometric_mean = exp(path_geometric_mean);

		// calculate payoff - save variables...
		path_arithmetic_mean = (direction == CALL ? max(path_arithmetic_mean - strike_price, 0.0f) : max(strike_price - path_arithmetic_mean, 0.0f));
		path_geometric_mean = (direction == CALL ? max(path_geometric_mean - adjusted_strike_price, 0.0f) : max(adjusted_strike_price - path_geometric_mean, 0.0f));

		// calculate discounted value
		path_arithmetic_mean *= discounting_factor;
		path_geometric_mean *= discounting_factor;


		// update sample statistics with new path value
		update_running_variance(&arithmetic_iterations, &running_arithmetic_mean, &running_arithmetic_m2, path_arithmetic_mean);
		update_running_variance(&geometric_iterations, &running_geometric_mean, &running_geometric_m2, path_geometric_mean);
		update_running_variance(&arithmetic_geometric_iterations, &running_arithmetic_geometric_mean, &running_arithmetic_geometric_m2, path_arithmetic_mean * path_geometric_mean);
	}

	arithmetic_results[tid].x = running_arithmetic_mean;
	arithmetic_results[tid].y = finalize_running_variance(&arithmetic_iterations, &running_arithmetic_mean, &running_arithmetic_m2);

	geometric_results[tid].x = running_geometric_mean;
	geometric_results[tid].y = finalize_running_variance(&geometric_iterations, &running_geometric_mean, &running_geometric_m2);

	arithmetic_geometric_means[tid] = running_arithmetic_geometric_mean;
}
