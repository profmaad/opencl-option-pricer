/* (C) 2014 Maximilian Gerhard Wolter */

# include <cstdlib>
# include <cstdio>
# include <cassert>

# include "kernels.h"
# include "matrix.h"
# include "json_helper.hpp"
# include "opencl_utils.hpp"

# include "basket_arithmetic_opencl_option.hpp"

BasketArithmeticOpenCLOption::BasketArithmeticOpenCLOption(JSONHelper &parameters) : MonteCarloOpenCLOption(parameters),
										     cl_start_prices(NULL),
										     cl_asset_volatilities(NULL),
										     cl_correlations(NULL),
										     cl_correlations_cholesky(NULL)
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

	correlations_cholesky = (float*)malloc(number_of_assets*number_of_assets*sizeof(float));
	assert(correlations_cholesky != NULL);
	
	cholesky_decomposition(number_of_assets, correlations, correlations_cholesky);
}
BasketArithmeticOpenCLOption::~BasketArithmeticOpenCLOption()
{
	free(start_prices);
	free(asset_volatilities);
	free(correlations);
	free(correlations_cholesky);

	if(cl_start_prices) { clfree(cl_start_prices); }
	if(cl_asset_volatilities) { clfree(cl_asset_volatilities); }
	if(cl_correlations) { clfree(cl_correlations); }
	if(cl_correlations_cholesky) { clfree(cl_correlations_cholesky); }
}

const char* BasketArithmeticOpenCLOption::kernel_symbol()
{
	return get_kernel_sym(Basket_Arithmetic, control_variate);
}

void BasketArithmeticOpenCLOption::setup_inputs()
{
	if(cl_start_prices) { clfree(cl_start_prices); }
	if(cl_asset_volatilities) { clfree(cl_asset_volatilities); }
	if(cl_correlations) { clfree(cl_correlations); }
	if(cl_correlations_cholesky) { clfree(cl_correlations_cholesky); }

	cl_start_prices = opencl_memcpy<cl_float, float>(context, number_of_assets, start_prices);
	cl_asset_volatilities = opencl_memcpy<cl_float, float>(context, number_of_assets, asset_volatilities);
	cl_correlations = opencl_memcpy<cl_float, float>(context, number_of_assets*number_of_assets, correlations);
	cl_correlations_cholesky = opencl_memcpy<cl_float, float>(context, number_of_assets*number_of_assets, correlations_cholesky);

	assert(cl_start_prices != NULL);
	assert(cl_asset_volatilities != NULL);
	assert(cl_correlations != NULL);
	assert(cl_correlations_cholesky != NULL);

	clmsync(context, device_number, cl_start_prices, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, device_number, cl_asset_volatilities, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, device_number, cl_correlations, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, device_number, cl_correlations_cholesky, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
}

void BasketArithmeticOpenCLOption::fork_kernel(cl_kernel kernel)
{
	assert(cl_start_prices != NULL);
	assert(cl_asset_volatilities != NULL);
	assert(cl_correlations != NULL);
	assert(cl_correlations_cholesky != NULL);

	clndrange_t index_range = clndrange_init1d(0, number_of_workers, number_of_workers);

	if(use_control_variate())
	{
		clforka(context, device_number, kernel, &index_range, CL_EVENT_NOWAIT, direction, number_of_assets, cl_start_prices, strike_price, maturity, cl_asset_volatilities, risk_free_rate, cl_correlations, cl_correlations_cholesky, number_of_paths, use_adjusted_strike(), seeds, arithmetic_results, geometric_results, arithmetic_geometric_means);
	}
	else
	{
		clforka(context, device_number, kernel, &index_range, CL_EVENT_NOWAIT, direction, number_of_assets, cl_start_prices, strike_price, maturity, cl_asset_volatilities, risk_free_rate, cl_correlations, cl_correlations_cholesky, number_of_paths, seeds, arithmetic_results);
	}
}

void BasketArithmeticOpenCLOption::cleanup()
{
	if(cl_start_prices) { clfree(cl_start_prices); }
	if(cl_asset_volatilities) { clfree(cl_asset_volatilities); }
	if(cl_correlations) { clfree(cl_correlations); }
	if(cl_correlations_cholesky) { clfree(cl_correlations_cholesky); }

	cl_start_prices = NULL;
	cl_asset_volatilities = NULL;
	cl_correlations = NULL;
	cl_correlations_cholesky = NULL;
}
