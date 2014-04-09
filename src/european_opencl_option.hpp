/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_EUROPEAN_OPENCL_OPTION_HPP
# define OCLOP_EUROPEAN_OPENCL_OPTION_HPP

# include <stdcl.h>

# include "json_helper.hpp"

# include "closed_form_opencl_option.hpp"

class EuropeanOpenCLOption : public ClosedFormOpenCLOption
{
public:
	EuropeanOpenCLOption(JSONHelper &parameters);

protected:
	const char* kernel_symbol();
	void setup_inputs() {}
	void fork_kernel(cl_kernel kernel);

	float start_price;
	float strike_price;
	float maturity;
	float volatility;
	float risk_free_rate;
};

# endif /*OCLOP_EUROPEAN_OPENCL_OPTION_HPP*/
