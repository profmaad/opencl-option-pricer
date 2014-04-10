/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <cassert>

# include <stdcl.h>

# include "json_helper.hpp"

# include "closed_form_opencl_option.hpp"

ClosedFormOpenCLOption::ClosedFormOpenCLOption(JSONHelper &parameters) : OpenCLOption(parameters),
	results(NULL)
{
}

void ClosedFormOpenCLOption::setup_outputs()
{
	results = (float*)clmalloc(context, 2*sizeof(float), 0);
	assert(results != NULL);

	clmsync(context, device_number, results, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
}

void ClosedFormOpenCLOption::retrieve_results()
{
	clmsync(context, device_number, results, CL_MEM_HOST|CL_EVENT_NOWAIT);
}

void ClosedFormOpenCLOption::finalize_results(float *population_mean, float *confidence_interval_lower, float *confidence_interval_upper)
{
	unsigned int result_index = 0;
	if(direction == Put) { result_index = 1; }

	*population_mean = results[result_index];

	*confidence_interval_lower = results[result_index];
	*confidence_interval_upper = results[result_index];
}

void ClosedFormOpenCLOption::cleanup()
{
	clfree(results);
	results = NULL;
}
