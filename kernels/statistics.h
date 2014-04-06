/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

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

void initialize_running_variance(unsigned int *iteration, float *a, float *q)
{
	*iteration = 0;
	*a = 0;
	*q = 0;
}
void update_running_variance(unsigned int *iteration, float *a, float *q, float sample)
{
	*iteration += 1;

	float new_a = *a + (sample - *a) / *iteration;

	float new_q = *q + (sample - *a) * (sample - new_a);

	*a = new_a;
	*q = new_q;
}
float finalize_running_variance(unsigned int *iteration, float *a, float *q)
{
	return *q / *iteration;
}
