/* (c) 2014 Maximilian Gerhard Wolter */

# include <random.h>
# include <statistics.h>


__kernel void random_test_host(unsigned int work_size, __global uint2 *seeds, __global float *results)
{
	int tid = get_global_id(0);

	uint2 seed = seeds[tid];

	uniform_int_prng_state base_state;
	initialize_uniform_int_prng(&base_state, seed);
	stdnormal_float_prng_state prng_state;
	initialize_stdnormal_float_prng(&prng_state, &base_state);

	unsigned long results_offset = tid*work_size;
	
	for(int i = 0; i < work_size; i++)
	{
		results[results_offset+i] = stdnormal_float_random(&prng_state);
	}
}

