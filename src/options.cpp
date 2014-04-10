/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <iostream>

# include "json_helper.hpp"

# include "options.hpp"

OpenCLOption* create_opencl_option(JSONHelper &parameters)
{
	OptionType type = parameters.get_type();

	switch(type)
	{
	case European:
		return new EuropeanOpenCLOption(parameters);
		break;
	case Asian_Geometric:
		return new AsianGeometricOpenCLOption(parameters);
		break;
	default:
		std::cerr << "[ERROR] Option type " << parameters.get_string("type") << " not yet implemented." << std::endl;
		return NULL;
		break;
	}
}
