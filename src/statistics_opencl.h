/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_STATISTICS_OPENCL_H
# define OCLOP_STATISTICS_OPENCL_H

# include <CL/opencl.h>

// this assumes that the layout for two-vectors is:
// vec.x -> (sub)sample mean
// vec.y -> (sub)sample variance

float population_mean_from_subsample_mean(const unsigned int number_of_samples, const cl_float2 *samples);

float population_variance_from_samples(const unsigned int number_of_samples, const cl_float2 *samples, const float population_mean);

void population_statistics_from_subsample_statistics(const unsigned int number_of_samples, const unsigned int sample_size, const cl_float2 *samples, float *population_mean, float *population_variance);

# endif /*OCLOP_STATISTICS_OPENCL_H*/
