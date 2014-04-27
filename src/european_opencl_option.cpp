/* (c) 2014 Maximilian Gerhard Wolter */

# include <cstdlib>
# include <cstdio>
# include <cassert>

# include "kernels.h"
# include "json_helper.hpp"

# include "european_opencl_option.hpp"

EuropeanOpenCLOption::EuropeanOpenCLOption(JSONHelper &parameters) : ClosedFormOpenCLOption(parameters)
{
	start_price = parameters.get_float("start_price");
	strike_price = parameters.get_float("strike_price");
	maturity = parameters.get_float("maturity");
	volatility = parameters.get_float("volatility");
	risk_free_rate = parameters.get_float("risk_free_rate");

	assert(start_price >= 0);
	assert(strike_price >= 0);
	assert(maturity > 0);
	assert(volatility >= 0);
	assert(risk_free_rate >= 0);
}

const char* EuropeanOpenCLOption::kernel_symbol()
{
	return get_kernel_sym(European, None);	
}

void EuropeanOpenCLOption::fork_kernel(cl_kernel kernel)
{
	clndrange_t index_range = clndrange_init1d(0, 1, 1);

	clforka(context, device_number, kernel, &index_range, CL_EVENT_NOWAIT, start_price, strike_price, maturity, volatility, risk_free_rate, results);
}
