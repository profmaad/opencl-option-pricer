/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_CLOSED_FORM_OPENCL_OPTION_HPP
# define OCLOP_CLOSED_FORM_OPENCL_OPTION_HPP

# include "opencl_option.hpp"

class ClosedFormOpenCLOption : public OpenCLOption
{
public:
	ClosedFormOpenCLOption();

protected:
	virtual void setup_inputs() = 0;
	void setup_outputs();
	virtual void fork_kernel(cl_kernel kernel) = 0;
	void retrieve_results();
	void finalize_results(float *population_mean, float *confidence_interval_lower, float *confidence_interval_upper);
	void cleanup();

	float *results;
};

# endif /*OCLOP_CLOSED_FORM_OPENCL_OPTION_HPP*/