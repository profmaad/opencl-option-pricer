/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_BASKET_ARITHMETIC_OPENCL_OPTION_HPP
# define OCLOP_BASKET_ARITHMETIC_OPENCL_OPTION_HPP

# include <stdcl.h>

# include "json_helper.hpp"

# include "monte_carlo_opencl_option.hpp"

class BasketArithmeticOpenCLOption : public MonteCarloOpenCLOption
{
public:
	BasketArithmeticOpenCLOption(JSONHelper &parameters);
	~BasketArithmeticOpenCLOption();

protected:
	const char* kernel_symbol();
	void setup_inputs();
	void fork_kernel(cl_kernel kernel);
	void cleanup();

	unsigned int number_of_assets;
	float *start_prices;
	float strike_price;
	float maturity;
	float *asset_volatilities;
	float risk_free_rate;
	float *correlations;
	float *correlations_cholesky;

	cl_float *cl_start_prices;
	cl_float *cl_asset_volatilities;
	cl_float *cl_correlations;
	cl_float *cl_correlations_cholesky;
};

# endif /*OCLOP_BASKET_ARITHMETIC_OPENCL_OPTION_HPP*/
