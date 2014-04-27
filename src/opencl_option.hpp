/* (c) 2014 Maximilian Gerhard Wolter */

# ifndef OCLOP_OPENCL_OPTION_HPP
# define OCLOP_OPENCL_OPTION_HPP

# include <CL/cl.h>
//# include <CL/opencl.h>
# include <stdcl.h>

# include "types.h"
# include "option.hpp"
# include "json_helper.hpp"

# include "opencl_utils.hpp"

class OpenCLOption : public Option
{
public:
	OpenCLOption(JSONHelper &parameters);
	virtual ~OpenCLOption();

	virtual void set_opencl_configuration(CLCONTEXT *context, unsigned int device_number, unsigned int number_of_workers);

	virtual void set_random_seeds(const random_seed *seeds);
	virtual void reset_random_seeds();

	bool price(float *population_mean, float *confidence_interval_lower, float *confidence_interval_upper);

protected:
	virtual const char* kernel_symbol() = 0;
	virtual void setup_inputs() = 0;
	virtual void setup_outputs() = 0;
	virtual void fork_kernel(cl_kernel kernel) = 0;
	virtual void retrieve_results() = 0;
	virtual void cleanup() = 0;
	virtual void finalize_results(float *population_mean, float *confidence_interval_lower, float *confidence_interval_upper) = 0;

	virtual void opencl_configuration_changed() {}

	CLCONTEXT *context;
	unsigned int device_number;
	unsigned int number_of_workers;

	random_seed *seeds;

	OptionDirection direction;
};

# endif /*OCLOP_OPENCL_OPTION_HPP*/
