/* (c) 2014 Maximilian Gerhard Wolter */

# include <random.h>
# include <statistics.h>

# include "geometric_asian.cl"

# define CALL 0
# define PUT 1

float arithmetic_asian_expected_underlying_price_at_maturity(float start_price, float risk_free_rate, float maturity, unsigned int averaging_steps)
{
	float factor = 1.0f/(float)averaging_steps * start_price;
	float delta_t = maturity/(float)averaging_steps;

	float sum = 0.0f;
	for(int i = 1; i <= averaging_steps; i++)
	{
		sum += factor * exp(risk_free_rate * i*delta_t);
	}

	return sum;
}

float arithmetic_asian_geometric_cv_adjusted_strike(float start_price, float strike_price, float maturity, float volatility, float risk_free_rate, unsigned int averaging_steps)
{
	float geometric_expectation = geometric_asian_expected_underlying_price_at_maturity(start_price, maturity, volatility, risk_free_rate, averaging_steps);
	float arithmetic_expectation = arithmetic_asian_expected_underlying_price_at_maturity(start_price, risk_free_rate, maturity, averaging_steps);

	return strike_price + geometric_expectation - arithmetic_expectation;
}

__kernel void arithmetic_asian_no_cv(unsigned int direction, float start_price, float strike_price, float maturity, float volatility, float risk_free_rate, unsigned int averaging_steps, unsigned int total_number_of_paths, __global uint2 *seeds, __global float2 *results)
{
	// get details on worker setup
	unsigned int tid = get_global_id(0);
	unsigned int worker_count = get_global_size(0);

	// calculate work item size
	unsigned int number_of_paths = total_number_of_paths/worker_count;

	// setup PRNG
	uint2 seed = seeds[tid];

	uniform_int_prng_state base_state;
	initialize_uniform_int_prng(&base_state, seed);
	stdnormal_float_prng_state prng_state;
	initialize_stdnormal_float_prng(&prng_state, &base_state);

	// setup running mean and variance calculation
	unsigned int statistics_iteration;
	float running_mean;
	float running_m2;
	initialize_running_variance(&statistics_iteration, &running_mean, &running_m2);
	
	// calculate fixed monte carlo parameters
	float delta_t = maturity/((float)averaging_steps);
	float sqrt_delta_t = sqrt(delta_t);
	float drift = exp((risk_free_rate - 0.5*volatility*volatility) * delta_t);
	float running_mean_sample_factor = 1.0f/(float)averaging_steps;
	float discounting_factor = exp(-risk_free_rate * maturity);

	float path_mean = 0;
	float asset_price = 0;
	float growth_factor = 0;

	for(int path = 0; path < number_of_paths; path++)
	{
		path_mean = 0;

		growth_factor = drift * exp(volatility * sqrt_delta_t * stdnormal_float_random(&prng_state));
		asset_price = start_price * growth_factor;
		path_mean += asset_price*running_mean_sample_factor;
		
		for(int i = 1; i < averaging_steps; i++)
		{
			growth_factor = drift * exp(volatility * sqrt_delta_t * stdnormal_float_random(&prng_state));
			asset_price *= growth_factor;
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


__kernel void arithmetic_asian_geometric_cv(unsigned int direction, float start_price, float strike_price, float maturity, float volatility, float risk_free_rate, unsigned int averaging_steps, unsigned int total_number_of_paths, unsigned int adjust_strike, __global uint2 *seeds, __global float2 *arithmetic_results, __global float2 *geometric_results, __global float *arithmetic_geometric_means)
{
	// get details on worker setup
	unsigned int tid = get_global_id(0);
	unsigned int worker_count = get_global_size(0);

	// calculate work item size
	unsigned int number_of_paths = total_number_of_paths/worker_count;

	// setup PRNG
	uint2 seed = seeds[tid];

	uniform_int_prng_state base_state;
	initialize_uniform_int_prng(&base_state, seed);
	stdnormal_float_prng_state prng_state;
	initialize_stdnormal_float_prng(&prng_state, &base_state);

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
	float delta_t = maturity/((float)averaging_steps);
	float drift = exp((risk_free_rate - 0.5*volatility*volatility) * delta_t);
	float path_mean_sample_factor = 1.0f/(float)averaging_steps;
	float discounting_factor = exp(-risk_free_rate * maturity);

	float adjusted_strike_price = (adjust_strike == 0 ? strike_price : arithmetic_asian_geometric_cv_adjusted_strike(start_price, strike_price, maturity, volatility, risk_free_rate, averaging_steps));

	for(int path = 0; path < number_of_paths; path++)
	{
		float path_arithmetic_mean = 0;
		float path_geometric_mean = 0;

		float growth_factor = drift * exp(volatility * sqrt(delta_t) * stdnormal_float_random(&prng_state));
		float asset_price = start_price;
		
		for(int i = 0; i < averaging_steps; i++)
		{
			growth_factor = drift * exp(volatility * sqrt(delta_t) * stdnormal_float_random(&prng_state));
			asset_price *= growth_factor;
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
