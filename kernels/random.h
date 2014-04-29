/* (c) 2014 Maximilian Gerhard Wolter */

# ifndef OPENCL_RANDOM_H
# define OPENCL_RANDOM_H

# include <mwc64x/mwc64x/mwc64x_rng.cl>

# ifndef M_PI_F
# define M_PI_F       3.14159274101257f
# endif

# ifndef UINT_MAX
# define UINT_MAX     0xffffffff
# endif

static const float INT_TO_FLOAT_FACTOR = 1.0/UINT_MAX;
static const float TWO_PI = 2 * M_PI_F;

typedef mwc64x_state_t uniform_int_prng_state;
typedef struct stdnormal_float_prng_state
{
	uniform_int_prng_state *int_prng_state;
	bool has_unused_value;
	float unused_value;
} stdnormal_float_prng_state;

typedef struct correlated_stdnormal_float_prng_state
{
	stdnormal_float_prng_state *base_state;
	unsigned int size;
	__global float *correlation_matrix;
	__global float *uncorrelated_random;
} correlated_stdnormal_float_prng_state;

void initialize_uniform_int_prng(uniform_int_prng_state *state, uint2 seed)
{
	state->x = seed.x;
	state->c = seed.y;
}
unsigned int uniform_uint_random(uniform_int_prng_state *state)
{
	return MWC64X_NextUint(state);
}

void initialize_stdnormal_float_prng(stdnormal_float_prng_state *state, uniform_int_prng_state *base_state)
{
	state->int_prng_state = base_state;
	state->has_unused_value = false;
	state->unused_value = 0.0f;
}
float stdnormal_float_random(stdnormal_float_prng_state *state)
{
	if(state->has_unused_value)
	{
		state->has_unused_value = false;
		return state->unused_value;
	}
	else
	{
		unsigned int uint_random1 = uniform_uint_random(state->int_prng_state);
		unsigned int uint_random2 = uniform_uint_random(state->int_prng_state);

		float u1 = (float)uint_random1*INT_TO_FLOAT_FACTOR;
		float u2 = (float)uint_random2*INT_TO_FLOAT_FACTOR;

		float r = sqrt(-2.0f * log(u1));

		float cosine;
		float sine;
		sincosf(TWO_PI * u2, &sine, &cosine);

		state->unused_value = r * sine;
		state->has_unused_value = true;

		return r * cosine;
	}
}

void initialize_correlated_stdnormal_float_prng(correlated_stdnormal_float_prng_state *state, stdnormal_float_prng_state *base_state, unsigned int size, __global float *correlations, __global float *uncorrelated_random)
{
	state->base_state = base_state;
	state->size = size;
	state->correlation_matrix = correlations;
	state->uncorrelated_random = uncorrelated_random;
}
void correlated_stdnormal_float_random(correlated_stdnormal_float_prng_state *state, __global float *random)
{
	// corr_rand = rand * correlations (vector = vector * matrix)

	for(int i = 0; i < state->size; i++)
	{
		state->uncorrelated_random[i] = stdnormal_float_random(state->base_state);
	}

	for(int i = 0; i < state->size; i++)
	{
		random[i] = 0.0f;
		for(int j = 0; j < state->size; j++)
		{
			random[i] += state->uncorrelated_random[j] * state->correlation_matrix[j*state->size + i];
		}
	}
}

# endif /*OPENCL_RANDOM_H*/
