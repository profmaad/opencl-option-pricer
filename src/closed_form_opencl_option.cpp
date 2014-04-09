/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <stdcl.h>

# include "closed_form_opencl_option.hpp"

ClosedFormOpenCLOption::ClosedFormOpenCLOption() : OpenCLOption(),
	results(NULL)
{
}

void ClosedFormOpenCLOption::setup_outputs()
{
	results = (float*)clmalloc(context, 2*sizeof(float), 0);
	clmsync(context, device_number, results, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
}

void ClosedFormOpenCLOption::retrieve_results()
{
	clmsync(context, device_number, results, CL_MEM_HOST|CL_EVENT_NOWAIT);
}

void ClosedFormOpenCLOption::finalize_results(float *population_mean, float *confidence_interval_lower, float *confidence_interval_upper)
{
	// TODO: check whether to do call or put

	*population_mean = results[0];

	*confidence_interval_lower = results[0];
	*confidence_interval_upper = results[0];
}

void ClosedFormOpenCLOption::cleanup()
{
	clfree(results);
	results = NULL;
}
