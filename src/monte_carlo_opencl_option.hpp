/* (c) 2014 Maximilian Gerhard Wolter */

# ifndef OCLOP_MONTE_CARLO_OPENCL_OPTION_HPP
# define OCLOP_MONTE_CARLO_OPENCL_OPTION_HPP

# include "types.h"
# include "json_helper.hpp"

# include "opencl_option.hpp"

class MonteCarloOpenCLOption : public OpenCLOption
{
public:
	MonteCarloOpenCLOption(JSONHelper &parameters);
	virtual ~MonteCarloOpenCLOption() {}

protected:
	virtual void setup_inputs() = 0;
	virtual void setup_outputs();
	virtual void fork_kernel(cl_kernel kernel) = 0;
	virtual void retrieve_results();
	virtual void finalize_results(float *population_mean, float *confidence_interval_lower, float *confidence_interval_upper);
	virtual void cleanup();

	bool use_control_variate();
	bool use_adjusted_strike();

	cl_float2 *arithmetic_results;
	cl_float2 *geometric_results;
	cl_float *arithmetic_geometric_means;

	ControlVariate control_variate;
	unsigned int number_of_paths;
	unsigned int paths_per_worker;
};

# endif /*OCLOP_MONTE_CARLO_OPENCL_OPTION_HPP*/
