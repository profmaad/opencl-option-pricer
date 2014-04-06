/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <Random123/threefry.h>

const float INT_TO_FLOAT_FACTOR = 1.0/INT_MAX;
const float TWO_PI = 2 * M_PI_F;

typedef struct uniform_int_prng_state
{
	union
	{
		threefry4x32_key_t key;
		uint4 seed1;
	} key;
	union
	{
		threefry4x32_ctr_t counter;
		uint4 seed2;
	} counter;
	union
	{
		threefry4x32_ctr_t ctr;
		uint uints[4];
	} random;
	unsigned int unused_values_remaining;
} uniform_int_prng_state;
typedef struct stdnormal_float_prng_state
{
	uniform_int_prng_state *int_prng_state;
	bool has_unused_value;
	float unused_value;
} stdnormal_float_prng_state;

void initialize_uniform_int_prng(uniform_int_prng_state *state, uint4 seed1, uint4 seed2)
{
	state->key.seed1 = seed1;
	state->counter.ctr = seed2;
	state->unused_values_remaining = 0;
}
unsigned int uniform_int_random(uniform_int_prng_state *state)
{
	if(state->unused_values_remaining > 0)
	{
		unsigned int rand = state->counter.uints[4-state->unused_values_remaining];
		state->unused_values_remaining--;

		return rand;
	}
	else
	{
		state->counter.counter.v[0]++;
		state->random.ctr = threefry4x32(state->counter.counter, state->key.key);
		state->unused_values_remaining = 3;

		return state->random.uints[0];
	}
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
		unsigned int uint_random1 = uniform_int_random(state->int_prng_state);
		unsigned int uint_random2 = uniform_int_random(state->int_prng_state);

		float u1 = (float)uint_random1*INT_TO_FLOAT_FACTOR;
		float u2 = (float)uint_random2*INT_TO_FLOAT_FACTOR;

		float r = sqrt(-2.0f * log(u1));

		float cosine;
		float sine = sincos(TWO_PI * u2, &cosine);

		state->unused_value = r * sine;
		state->has_unused_value = true;

		return r * cosine;
	}
}
