/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

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

# define DEFAULT_CONTEXT ((stdgpu)? stdgpu : stdcpu)
# define DEVICE_NUMBER 0
# define NUMBER_OF_WORKERS 16

JSONHelper* read_json_from_stdin()
{
	std::stringstream json_source;

	for(std::string line; std::getline(std::cin, line);)
	{
		json_source << line << std::endl;
	}
	std::cerr << "JSON INPUT------------------" << std::endl;
	std::cerr << json_source.str();
	std::cerr << "JSON END--------------------" << std::endl;

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
	CLCONTEXT* context = DEFAULT_CONTEXT;
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
	option->reset_random_seeds();

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
