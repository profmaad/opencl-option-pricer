/* (C) 2014 Maximilian Gerhard Wolter */

# include <cstdlib>
# include <cstdio>
# include <cassert>

# include "kernels.h"
# include "json_helper.hpp"
# include "opencl_utils.hpp"

# include "basket_geometric_opencl_option.hpp"

BasketGeometricOpenCLOption::BasketGeometricOpenCLOption(JSONHelper &parameters) : ClosedFormOpenCLOption(parameters),
										   cl_start_prices(NULL),
										   cl_asset_volatilities(NULL),
										   cl_correlations(NULL)
{	
	unsigned int tmp_size;

	strike_price = parameters.get_float("strike_price");
	maturity = parameters.get_float("maturity");
	risk_free_rate = parameters.get_float("risk_free_rate");

	start_prices = parameters.get_vector("start_price", &number_of_assets);
	asset_volatilities = parameters.get_vector("volatility", &tmp_size);
	if(tmp_size < number_of_assets) { number_of_assets = tmp_size; }
	correlations = parameters.get_matrix("correlation", &tmp_size);
	if(tmp_size < number_of_assets) { number_of_assets = tmp_size; }

	assert(start_prices != NULL);
	assert(asset_volatilities != NULL);
	assert(correlations != NULL);

	assert(number_of_assets > 0);

	assert(strike_price >= 0);
	assert(maturity > 0);
	assert(risk_free_rate >= 0);

	for(unsigned int i = 0; i < number_of_assets; i++)
	{
		assert(start_prices[i] >= 0);
		assert(asset_volatilities[i] >= 0);

		for(unsigned int j = 0; j < number_of_assets; j++)
		{
			assert(correlations[i*number_of_assets + j] >= 0);
			assert(correlations[i*number_of_assets + j] <= 1);
		}
	}
}
BasketGeometricOpenCLOption::~BasketGeometricOpenCLOption()
{
	free(start_prices);
	free(asset_volatilities);
	free(correlations);

	if(cl_start_prices) { clfree(cl_start_prices); }
	if(cl_asset_volatilities) { clfree(cl_asset_volatilities); }
	if(cl_correlations) { clfree(cl_correlations); }
}

const char* BasketGeometricOpenCLOption::kernel_symbol()
{
	return get_kernel_sym(Basket_Geometric, None);	
}

void BasketGeometricOpenCLOption::setup_inputs()
{
	if(cl_start_prices) { clfree(cl_start_prices); }
	if(cl_asset_volatilities) { clfree(cl_asset_volatilities); }
	if(cl_correlations) { clfree(cl_correlations); }

	cl_start_prices = opencl_memcpy<cl_float, float>(context, number_of_assets, start_prices);
	cl_asset_volatilities = opencl_memcpy<cl_float, float>(context, number_of_assets, asset_volatilities);
	cl_correlations = opencl_memcpy<cl_float, float>(context, number_of_assets*number_of_assets, correlations);

	assert(cl_start_prices != NULL);
	assert(cl_asset_volatilities != NULL);
	assert(cl_correlations != NULL);

	clmsync(context, device_number, cl_start_prices, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, device_number, cl_asset_volatilities, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, device_number, cl_correlations, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
}

void BasketGeometricOpenCLOption::fork_kernel(cl_kernel kernel)
{
	assert(cl_start_prices != NULL);
	assert(cl_asset_volatilities != NULL);
	assert(cl_correlations != NULL);

	clndrange_t index_range = clndrange_init1d(0, 1, 1);

	clforka(context, device_number, kernel, &index_range, CL_EVENT_NOWAIT, number_of_assets, cl_start_prices, strike_price, maturity, cl_asset_volatilities, risk_free_rate, cl_correlations, results);
}

void BasketGeometricOpenCLOption::cleanup()
{
	if(cl_start_prices) { clfree(cl_start_prices); }
	if(cl_asset_volatilities) { clfree(cl_asset_volatilities); }
	if(cl_correlations) { clfree(cl_correlations); }

	cl_start_prices = NULL;
	cl_asset_volatilities = NULL;
	cl_correlations = NULL;
}
