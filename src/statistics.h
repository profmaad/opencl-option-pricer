/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_STATISTICS_H
# define OCLOP_STATISTICS_H

float population_mean_from_subsample_mean(const unsigned int number_of_samples, const float *samples);

float population_variance_from_samples(const unsigned int number_of_samples, const float *samples, const float population_mean);

void confidence_interval_95percent(const unsigned int number_of_samples, const float mean, const float variance, float *lower, float *upper);

# endif /*OCLOP_STATISTICS_H*/
