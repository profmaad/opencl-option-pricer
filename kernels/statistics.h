/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OPENCL_STATISTICS_H
# define OPENCL_STATISTICS_H

float arithmetic_mean(unsigned int n, float *values)
{
	float mean = 0.0;

	for(int i = 0; i < n; i++)
	{
		mean += values[i]/n;
	}

	return mean;
}
float geometric_mean(unsigned int n, float *values)
{
	float sum = 0.0;

	for(int i = 0; i < n; i++)
	{
		sum += log(values[i])/n;
	}

	return exp(sum);
}

void initialize_running_variance(unsigned int *iteration, float *mean, float *m2)
{
	*iteration = 0;
	*mean = 0;
	*m2 = 0;
}
void update_running_variance(unsigned int *iteration, float *mean, float *m2, float sample)
{
	*iteration += 1;

	float new_mean = *mean + (sample - *mean) / *iteration;

	float new_m2 = *m2 + (sample - *mean) * (sample - new_mean);

	*mean = new_mean;
	*m2 = new_m2;
}
float finalize_running_variance(unsigned int *iteration, float *mean, float *m2)
{
	return *m2 / ((*iteration)-1);
}

# endif /*OPENCL_STATISTICS_H*/
