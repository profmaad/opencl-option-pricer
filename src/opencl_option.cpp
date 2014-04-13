/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <iostream>

# include <cstddef>
# include <cassert>

# include <CL/cl.h>
//# include <CL/opencl.h>

# include "opencl_utils.hpp"
# include "json_helper.hpp"

# include "opencl_option.hpp"

OpenCLOption::OpenCLOption(JSONHelper &parameters) : Option(),
	context(NULL),
	device_number(0),
	number_of_workers(1),
	seeds(NULL)
{
	direction = parameters.get_direction();
}
OpenCLOption::~OpenCLOption()
{
	if(seeds) { clfree(seeds); }
}

void OpenCLOption::set_opencl_configuration(CLCONTEXT *context, unsigned int device_number, unsigned int number_of_workers)
{
	this->context = context;
	this->device_number = device_number;
	this->number_of_workers = number_of_workers;
}

void OpenCLOption::set_random_seeds(const random_seed *seeds)
{
	assert(this->context != NULL);
	assert(this->number_of_workers > 0);

	if(this->seeds != NULL)
	{
		clfree(this->seeds);
	}
	this->seeds = opencl_memcpy<random_seed, random_seed>(context, number_of_workers, seeds);

	assert(this->seeds != NULL);
}
void OpenCLOption::reset_random_seeds()
{
	assert(this->context != NULL);
	assert(this->number_of_workers > 0);

	if(this->seeds != NULL)
	{
		clfree(this->seeds);
	}
	
	this->seeds = generate_seeds(context, number_of_workers);

	assert(this->seeds != NULL);
}

bool OpenCLOption::price(float *population_mean, float *confidence_interval_lower, float *confidence_interval_upper)
{
	assert(this->context != NULL);
	assert(this->number_of_workers > 0);
	assert(this->seeds != NULL);

	assert(population_mean != NULL);
	assert(confidence_interval_lower != NULL);
	assert(confidence_interval_upper != NULL);

	const char *kernel_sym = kernel_symbol();
	if(!kernel_sym)
	{
		std::cerr << "[ERROR] failed to get kernel symbol" << std::endl;
		return false;
	}

	cl_kernel kernel = clsym(context, NULL, kernel_sym, 0);
	if(!kernel)
	{
		std::cerr << "[ERROR] failed to get kernel for symbol " << kernel_sym << std::endl;
		return false;
	}

	// setup input and output variables, sync them to the device
	clmsync(context, device_number, seeds, CL_MEM_DEVICE|CL_EVENT_NOWAIT);

	setup_inputs();
	setup_outputs();

	fork_kernel(kernel);
	
	retrieve_results();

	// force execution of operations in command queue (non-blocking call)
	clflush(context, device_number, 0);

	// block on completion of operations in command queue
	clwait(context, device_number, CL_ALL_EVENT);

	finalize_results(population_mean, confidence_interval_lower, confidence_interval_upper);

	cleanup();
	//if(kernel) { clclose(context, kernel); }

	return true;
}
