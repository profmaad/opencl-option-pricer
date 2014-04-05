/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

float geometric_mean(unsigned int n, float *values)
{
	float sum = 0.0;

	for(int i = 0; i < n; i++)
	{
		sum += log(values[i])/n;
	}

	return exp(sum);
}
