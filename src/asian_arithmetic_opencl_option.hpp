/* (c) 2014 Maximilian Gerhard Wolter */

# ifndef OCLOP_ASIAN_ARITHMETIC_OPENCL_OPTION_HPP
# define OCLOP_ASIAN_ARITHMETIC_OPENCL_OPTION_HPP

# include <stdcl.h>

# include "json_helper.hpp"

# include "monte_carlo_opencl_option.hpp"

class AsianArithmeticOpenCLOption : public MonteCarloOpenCLOption
{
public:
	AsianArithmeticOpenCLOption(JSONHelper &parameters);

protected:
	const char* kernel_symbol();
	void setup_inputs() {}
	void fork_kernel(cl_kernel kernel);

	float start_price;
	float strike_price;
	float maturity;
	float volatility;
	float risk_free_rate;
	unsigned int averaging_steps;
};

# endif /*OCLOP_ASIAN_ARITHMETIC_OPENCL_OPTION_HPP*/
