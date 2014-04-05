/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include "stdnormal.h"

float geometric_asian_volatility(float original_volatility, float steps)
{
	float volatility = original_volatility*sqrt(((steps+1)*(2*steps+1))/(6*steps*steps));

	return volatility;
}
float geometric_asian_mu(float risk_free_rate, float steps, float original_volatility, float volatility)
{
	float mu = (risk_free_rate - 0.5 * original_volatility*original_volatility)*((steps+1)/(2*steps)) + 0.5 * volatility*volatility;

	return mu;
}

float geometric_asian_d1(float start_price, float strike_price, float maturity, float volatility, float mu)
{
	float d1 = (log(start_price/strike_price) + (mu + 0.5 * volatility*volatility) * maturity)/(volatility * sqrt(maturity));

	return d1;
}
float geometric_asian_d2(float d1, float volatility, float maturity)
{
	float d2 = d1 - volatility * sqrt(maturity);

	return d2;
}

__kernel void geometric_asian(float start_price, float strike_price, float maturity, float original_volatility, float risk_free_rate, float steps, __global float *prices)
{
	float volatility = geometric_asian_volatility(original_volatility, steps);
	float mu = geometric_asian_mu(risk_free_rate, steps, original_volatility, volatility);

	float d1 = geometric_asian_d1(start_price, strike_price, maturity, volatility, mu);
	float d2 = geometric_asian_d2(d1, volatility, maturity);

	float discounting_factor = exp(-risk_free_rate * maturity);

	prices[0] = discounting_factor * (start_price * exp(mu * maturity) * stdnormal_cdf(d1) - strike_price * stdnormal_cdf(d2));
	prices[1] = discounting_factor * (strike_price * stdnormal_cdf(-d2) - start_price * exp(mu * maturity) * stdnormal_cdf(-d1));
}

