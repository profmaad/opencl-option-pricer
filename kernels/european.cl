/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include "stdnormal.h"

float european_d1(float start_price, float strike_price, float maturity, float volatility, float risk_free_rate)
{
	float d1 = (log(start_price/strike_price) + risk_free_rate*maturity)/(volatility * sqrt(maturity)) + 0.5*volatility*sqrt(maturity);

	return d1;
}
float european_d2(float d1, float volatility, float maturity)
{
	float d2 = d1 - volatility*sqrt(maturity);

	return d2;
}

__kernel void european(__global float start_price, __global float strike_price, __global float maturity, __global float volatility, __global float risk_free_rate, __global float *prices)
{
	float d1 = european_d1(start_price, strike_price, maturity, volatility, risk_free_rate);
	float d2 = european_d2(d1, volatility, maturity);

	prices[0] = start_price * stdnormal_cdf(d1) - strike_price * exp(-risk_free_rate*maturity) * stdnormal_cdf(d2);
	prices[1] = strike_price * exp(-risk_free_rate*maturity) * stdnormal_cdf(-d2) - start_price * stdnormal_cdf(-d1);
}
