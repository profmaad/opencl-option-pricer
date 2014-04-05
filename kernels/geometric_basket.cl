/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include "stdnormal.h"
# include "means.h"

float geometric_basket_volatility(unsigned int number_of_assets, float *asset_volatilities, float *correlations)
{
	float sum = 0.0;

	for(int i = 0; i < number_of_assets; i++)
	{
		for(int j = 0; j < number_of_assets; j++)
		{
			sum += asset_volatilities[i] * asset_volatilities[j] * correlations[i*number_of_assets + j];
		}
	}

	float volatility = sqrt(sum)/number_of_assets;

	return volatility;
}
float geometric_basket_mu(unsigned int number_of_assets, float risk_free_rate, float *asset_volatilities, float volatility)
{
	float sum = 0.0;

	for (int i = 0; i < number_of_assets; i++)
	{
		sum += (asset_volatilities[i]*asset_volatilities[i])/(2*number_of_assets);
	}

	float mu = risk_free_rate - sum + 0.5 * volatility*volatility;

	return mu;
}

float geometric_basket_start_price(unsigned int number_of_assets, float *start_prices)
{
	return geometric_mean(number_of_assets, start_prices);
}

float geometric_basket_d1(float start_price, float strike_price, float maturity, float volatility, float mu)
{
	float d1 = (log(start_price/strike_price) + (mu + 0.5 * volatility*volatility) * maturity)/(volatility * sqrt(maturity));

	return d1;
}
float geometric_basket_d2(float d1, float volatility, float maturity)
{
	float d2 = d1 - volatility * sqrt(maturity);

	return d2;
}

__kernel void geometric_basket(unsigned int number_of_assets, __global float *start_prices, float strike_price, float maturity, __global float *asset_volatilities, float risk_free_rate, __global float *correlations , __global float *prices)
{
	float volatility = geometric_basket_volatility(number_of_assets, asset_volatilities, correlations);
	float mu = geometric_basket_mu(number_of_assets, risk_free_rate, asset_volatilities, volatility);
	float start_price = geometric_basket_start_price(number_of_assets, start_prices);

	float d1 = geometric_basket_d1(start_price, strike_price, maturity, volatility, mu);
	float d2 = geometric_basket_d2(d1, volatility, maturity);

	float discounting_factor = exp(-risk_free_rate * maturity);

	prices[0] = discounting_factor * (start_price * exp(mu * maturity) * stdnormal_cdf(d1) - strike_price * stdnormal_cdf(d2));
	prices[1] = discounting_factor * (strike_price * stdnormal_cdf(-d2) - start_price * exp(mu * maturity) * stdnormal_cdf(-d1));
}

