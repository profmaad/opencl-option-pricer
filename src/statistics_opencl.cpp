/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include "statistics_opencl.h"

float population_mean_from_subsample_mean(const unsigned int number_of_samples, const cl_float2 *samples)
{
	float result = 0.0f;
	float factor = 1.0f/number_of_samples;
	
	for(int i = 0; i < number_of_samples; i++)
	{
		result += samples[i].x * factor;
	}

	return result;
}

float population_variance_from_samples(const unsigned int number_of_samples, const cl_float2 *samples, const float population_mean)
{
	float result = 0.0f;
	float factor = 1.0f/number_of_samples;

	for(int i = 0; i < number_of_samples; i++)
	{
		result += (samples[i].x - population_mean) * (samples[i].x - population_mean) * factor;
	}

	return result;
}

void population_statistics_from_subsample_statistics(const unsigned int number_of_samples, const unsigned int sample_size, const cl_float2 *samples, float *population_mean, float *population_variance)
{
	float factor = 1.0f/number_of_samples;

	*population_mean = population_mean_from_subsample_mean(number_of_samples, samples);
	
	float variance_of_sample_means = population_variance_from_samples(number_of_samples, samples, *population_mean);

	float sum_of_sample_variances = 0.0f;
	for(int i = 0; i < number_of_samples; i++)
	{
		sum_of_sample_variances += samples[i].y;
	}

	*population_variance =
		(sample_size - 1.0f) / (number_of_samples*sample_size - 1.0f) *
		(
			sum_of_sample_variances + 
			(
				(sample_size * (number_of_samples - 1.0f)) / (sample_size-1.0f)
				* variance_of_sample_means
			)
		)
		;
}
