/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <random.h>
# include <statistics.h>


__kernel void random_test(unsigned int work_size, __global uint2 *seeds, __global float2 *results)
{
	int tid = get_global_id(0);

	uint2 seed = seeds[tid];

	uniform_int_prng_state base_state;
	initialize_uniform_int_prng(&base_state, seed);
	stdnormal_float_prng_state prng_state;
	initialize_stdnormal_float_prng(&prng_state, &base_state);
	
	unsigned int variance_iteration;
	float variance_a;
	float variance_q;

	initialize_running_variance(&variance_iteration, &variance_a, &variance_q);

	for(int i = 0; i < work_size; i++)
	{
		float sample = stdnormal_float_random(&prng_state);

		update_running_variance(&variance_iteration, &variance_a, &variance_q, sample);
	}

	float sample_variance = finalize_running_variance(&variance_iteration, &variance_a, &variance_q);
	float sample_mean = variance_a;

	results[tid].x = sample_mean;
	results[tid].y = sample_variance;
}

