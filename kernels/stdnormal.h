// taken from NVidia Black Scholes OpenCL example
// here: https://developer.nvidia.com/opencl
float stdnormal_cdf(float x)
{
	const float       A1 = 0.31938153f;
	const float       A2 = -0.356563782f;
	const float       A3 = 1.781477937f;
	const float       A4 = -1.821255978f;
	const float       A5 = 1.330274429f;
	const float RSQRT2PI = 0.39894228040143267793994605993438f;

	float K = 1.0f / (1.0f + 0.2316419f * fabs(x));

	float cdf = RSQRT2PI * exp(- 0.5f * x * x) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

	if(x > 0)
	{
		cdf = 1.0f - cdf;
	}

	return cdf;
}
