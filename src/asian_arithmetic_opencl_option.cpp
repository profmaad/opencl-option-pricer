/* (C) 2014 Maximilian Gerhard Wolter */

# include <cstdlib>
# include <cstdio>
# include <cassert>

# include "kernels.h"
# include "json_helper.hpp"

# include "asian_arithmetic_opencl_option.hpp"

AsianArithmeticOpenCLOption::AsianArithmeticOpenCLOption(JSONHelper &parameters) : MonteCarloOpenCLOption(parameters)
{
	start_price = parameters.get_float("start_price");
	strike_price = parameters.get_float("strike_price");
	maturity = parameters.get_float("maturity");
	volatility = parameters.get_float("volatility");
	risk_free_rate = parameters.get_float("risk_free_rate");
	averaging_steps = parameters.get_uint("averaging_steps");

	assert(start_price >= 0);
	assert(strike_price >= 0);
	assert(maturity > 0);
	assert(volatility >= 0);
	assert(volatility <= 1);
	assert(risk_free_rate >= 0);
	assert(risk_free_rate <= 1);     
	assert(averaging_steps > 0);
}

const char* AsianArithmeticOpenCLOption::kernel_symbol()
{
	return get_kernel_sym(Asian_Arithmetic, control_variate);
}

void AsianArithmeticOpenCLOption::fork_kernel(cl_kernel kernel)
{
	clndrange_t index_range = clndrange_init1d(0, number_of_workers, number_of_workers);

	if(use_control_variate())
	{
		clforka(context, device_number, kernel, &index_range, CL_EVENT_NOWAIT, direction, start_price, strike_price, maturity, volatility, risk_free_rate, averaging_steps, number_of_paths, use_adjusted_strike(), seeds, arithmetic_results, geometric_results, arithmetic_geometric_means);
	}
	else
	{
		clforka(context, device_number, kernel, &index_range, CL_EVENT_NOWAIT, direction, start_price, strike_price, maturity, volatility, risk_free_rate, averaging_steps, number_of_paths, seeds, arithmetic_results);
	}
}
