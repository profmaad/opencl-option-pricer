/* (c) 2014 Maximilian Gerhard Wolter */

# include <cmath>

# include "statistics.h"

float population_mean_from_subsample_mean(const unsigned int number_of_samples, const float *samples)
{
	float result = 0.0f;
	float factor = 1.0f/number_of_samples;
	
	for(int i = 0; i < number_of_samples; i++)
	{
		result += samples[i] * factor;
	}

	return result;
}

float population_variance_from_samples(const unsigned int number_of_samples, const float *samples, const float population_mean)
{
	float result = 0.0f;
	float factor = 1.0f/number_of_samples;

	for(int i = 0; i < number_of_samples; i++)
	{
		result += (samples[i] - population_mean) * (samples[i] - population_mean) * factor;
	}

	return result;
}

void confidence_interval_95percent(const unsigned int number_of_samples, const float mean, const float variance, float *lower, float *upper)
{
	float offset = 1.96 * (sqrt(variance)/sqrt(number_of_samples));

	*lower = mean - offset;
	*upper = mean + offset;
}
