/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <random.h>
# include <statistics.h>

// results (no cv):
// arithmetic mean (running per path)
// -> payoff (per path)
//    -> value (per path)
//       -> value mean (running total)
//       -> value stddev (running total)

# define CALL 0
# define PUT 1

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
	float drift = exp((risk_free_rate - 0.5*volatility*volatility) * delta_t);
	float running_mean_sample_factor = 1.0f/(float)averaging_steps;
	float discounting_factor = exp(-risk_free_rate * maturity);

	for(int path = 0; path < number_of_paths; path++)
	{
		float path_mean = 0;

		float growth_factor = drift * exp(volatility * sqrt(delta_t) * stdnormal_float_random(&prng_state));
		float asset_price = start_price * growth_factor;
		path_mean += asset_price*running_mean_sample_factor;
		
		for(int i = 1; i < averaging_steps; i++)
		{
			growth_factor = drift * exp(volatility * sqrt(delta_t) * stdnormal_float_random(&prng_state));
			asset_price *= growth_factor;
			path_mean += asset_price*running_mean_sample_factor;
		}

		// calculate payoff - save variables...
		path_mean = (direction == CALL ? max(path_mean - strike_price, 0) : max(strike_price - path_mean, 0));
		// calculate discounted value
		path_mean *= discounting_factor;


		// update sample statistics with new path value
		update_running_variance(&statistics_iteration, &running_mean, &running_m2, path_mean);
	}

	results[tid].x = running_mean;
	results[tid].y = finalize_running_variance(&statistics_iteration, &running_mean, &running_m2);
}
