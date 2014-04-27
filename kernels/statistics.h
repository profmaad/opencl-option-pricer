/* (c) 2014 Maximilian Gerhard Wolter */

# ifndef OPENCL_STATISTICS_H
# define OPENCL_STATISTICS_H

float arithmetic_mean(unsigned int n, float *values)
{
	float mean = 0.0;

	float factor = 1.0f/(float)n;

	for(int i = 0; i < n; i++)
	{
		mean += values[i] * factor;
	}

	return mean;
}
float geometric_mean(unsigned int n, float *values)
{
	float sum = 0.0;

	float factor = 1.0f/(float)n;

	for(int i = 0; i < n; i++)
	{
		sum += log(values[i]) * factor;
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
	if(*iteration == 1) { return 0; }
	return *m2 / ((*iteration)-1);
}

# endif /*OPENCL_STATISTICS_H*/
