/* (c) 2014 Maximilian Gerhard Wolter */

# include <iostream>
# include <string>
# include <sstream>

# include <cstdio>
# include <cstdlib>
# include <ctime>
# include <cmath>

# include <stdcl.h>

# include "options.hpp"
# include "json_helper.hpp"
# include "json_outputter.hpp"

# define DEFAULT_CONTEXT ((stdacc)? stdacc : stdcpu)
# define DEVICE_NUMBER 0
# define NUMBER_OF_WORKERS 16

# define FIXED_SRAND_SEED 0xCAFE2342
# define USE_FIXED_SEEDS true

JSONHelper* read_json_from_stdin()
{
	std::stringstream json_source;

	for(std::string line; std::getline(std::cin, line);)
	{
		json_source << line << std::endl;
	}

	return new JSONHelper(json_source.str());
}

int main(int argc, char **argv)
{
	JSONHelper *json_helper = read_json_from_stdin();
	if(!json_helper->is_good())
	{
		std::cerr << "[ERROR] Failed to read input parameters from STDIN, or input wasn't valid JSON." << std::endl;
		std::cerr << json_helper->get_parsing_errors() << std::endl;
		return 1;
	}

	srand(time(NULL));

	stdcl_init();

	/* use default contexts, if no GPU use CPU */
        CLCONTEXT* context = NULL;
        if(stdacc)
	{
		std::cerr << "[INFO] using Epiphany accelerator" << std::endl;
		context = stdacc;
	}
        else if(stdgpu)
	{
		std::cerr << "[INFO] using GPU" << std::endl;
		context = stdgpu;
	}
        else
	{
		std::cerr << "[WARN] falling back to CPU" << std::endl;
		context = stdcpu;
	}
	if(!context)
	{
		std::cerr << "[ERROR] No valid OpenCL context found." << std::endl;
		return 2;
	}
	unsigned int device_number = DEVICE_NUMBER;
	unsigned int number_of_workers = NUMBER_OF_WORKERS;

	clopen(context, NULL, CLLD_NOW);

	OpenCLOption *option = create_opencl_option(*json_helper);
	if(!option)
	{
		std::cerr << "[ERROR] Failed to create a valid option from input parameters." << std::endl;
		return 2;
	}

	option->set_opencl_configuration(context, device_number, number_of_workers);

	// generate fixed random seeds based on fixed srand seed given above
	if(USE_FIXED_SEEDS)
	{
		random_seed fixed_seeds[number_of_workers];
		srand(FIXED_SRAND_SEED);
		for(int i = 0; i < number_of_workers; i++)
		{
			fixed_seeds[i].x = rand();
			fixed_seeds[i].y = rand();
		}
		option->set_random_seeds(fixed_seeds);
	}
	else
	{
		option->reset_random_seeds();
	}

	float mean;
	float confidence_interval_lower, confidence_interval_upper;

	option->price(&mean, &confidence_interval_lower, &confidence_interval_upper);
	delete option;
	delete json_helper;

	fprintf(stderr, "\tMean:      %10.5f\n", mean);
	fprintf(stderr, "\tCI:      [ %10.7f,\n\t           %10.7f ]\n", confidence_interval_lower, confidence_interval_upper);
	fprintf(stderr, "\tCI size:   %10.7f\n", confidence_interval_upper-confidence_interval_lower);

	JSONOutputter outputter(mean, confidence_interval_lower, confidence_interval_upper);
	std::cout << outputter.output() << std::endl;

	return 0;
}
