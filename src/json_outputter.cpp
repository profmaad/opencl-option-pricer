/* (c) 2014 Maximilian Gerhard Wolter */

# include <string>

# include <cmath>

# include <json/json.h>

# include "json_outputter.hpp"

JSONOutputter::JSONOutputter(float mean, float confidence_interval_lower, float confidence_interval_upper) :
	mean(mean),
	confidence_interval_lower(confidence_interval_lower),
	confidence_interval_upper(confidence_interval_upper),
	number_of_paths(0)
{	
}

void JSONOutputter::set_number_of_paths(unsigned int number_of_paths)
{
	this->number_of_paths = number_of_paths;
}

std::string JSONOutputter::output()
{
	Json::Value root;

	Json::Value json_confidence_interval;
	if(std::isnormal(confidence_interval_lower) && std::isnormal(confidence_interval_upper))
	{
		json_confidence_interval.append(confidence_interval_lower);
		json_confidence_interval.append(confidence_interval_upper);
	}
	else
	{
		json_confidence_interval.append(-1.0f);
		json_confidence_interval.append(-1.0f);
	}

	if(std::isnormal(mean)) { root["mean"] = mean; }
	else { root["mean"] = -1.0f; }

	root["confidence_interval"] = json_confidence_interval;

	if(number_of_paths > 0)
	{
		root["paths"] = number_of_paths;
	}

	Json::StyledWriter writer;
	return writer.write(root);
}
