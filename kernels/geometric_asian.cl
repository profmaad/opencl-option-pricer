/* (c) 2014 Maximilian Gerhard Wolter */

# include "stdnormal.h"

float geometric_asian_volatility(float original_volatility, unsigned int averaging_steps)
{
	float volatility = original_volatility*sqrt(((averaging_steps+1.0f)*(2*averaging_steps+1.0f))/(6.0f*averaging_steps*averaging_steps));

	return volatility;
}
float geometric_asian_mu(float risk_free_rate, unsigned int averaging_steps, float original_volatility, float volatility)
{
	float mu = (risk_free_rate - 0.5 * original_volatility*original_volatility)*((averaging_steps+1.0f)/(2.0f*averaging_steps)) + 0.5 * volatility*volatility;

	return mu;
}

float geometric_asian_expected_underlying_price_at_maturity(float start_price, float maturity, float original_volatility, float risk_free_rate, unsigned int averaging_steps)
{
	float volatility = geometric_asian_volatility(original_volatility, averaging_steps);
	float mu = geometric_asian_mu(risk_free_rate, averaging_steps, original_volatility, volatility);

	float price = exp(mu * maturity) * start_price;

	return price;
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

__kernel void geometric_asian(float start_price, float strike_price, float maturity, float original_volatility, float risk_free_rate, unsigned int averaging_steps, __global float *prices)
{
	float volatility = geometric_asian_volatility(original_volatility, averaging_steps);
	float mu = geometric_asian_mu(risk_free_rate, averaging_steps, original_volatility, volatility);

	float d1 = geometric_asian_d1(start_price, strike_price, maturity, volatility, mu);
	float d2 = geometric_asian_d2(d1, volatility, maturity);

	float discounting_factor = exp(-risk_free_rate * maturity);

	prices[0] = discounting_factor * (start_price * exp(mu * maturity) * stdnormal_cdf(d1) - strike_price * stdnormal_cdf(d2));
	prices[1] = discounting_factor * (strike_price * stdnormal_cdf(-d2) - start_price * exp(mu * maturity) * stdnormal_cdf(-d1));
}

