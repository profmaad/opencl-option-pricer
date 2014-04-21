/* (c) 2014 Maximilian Gerhard Wolter */

# ifndef OCLOP_BASKET_GEOMETRIC_OPENCL_OPTION_HPP
# define OCLOP_BASKET_GEOMETRIC_OPENCL_OPTION_HPP

# include <stdcl.h>

# include "json_helper.hpp"

# include "closed_form_opencl_option.hpp"

class BasketGeometricOpenCLOption : public ClosedFormOpenCLOption
{
public:
	BasketGeometricOpenCLOption(JSONHelper &parameters);
	~BasketGeometricOpenCLOption();

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

	cl_float *cl_start_prices;
	cl_float *cl_asset_volatilities;
	cl_float *cl_correlations;
};

# endif /*OCLOP_BASKET_GEOMETRIC_OPENCL_OPTION_HPP*/
