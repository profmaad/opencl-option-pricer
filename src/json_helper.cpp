/* (c) 2014 Maximilian Gerhard Wolter */

# include <string>

# include <cstdlib>
# include <cassert>

# include <json/json.h>

# include "json_helper.hpp"

JSONHelper::JSONHelper(const std::string &json_source)
{
	parsing_successful = reader.parse(json_source, root);
}

bool JSONHelper::is_good()
{
	return parsing_successful && (root.isObject()) ;
}
std::string JSONHelper::get_parsing_errors()
{
	return reader.getFormatedErrorMessages();
}

float JSONHelper::get_float(const std::string &key)
{
	return (float)root.get(key, 0.0f).asDouble();
}
unsigned int JSONHelper::get_uint(const std::string &key)
{
	return root.get(key, 0).asUInt();
}
const std::string JSONHelper::get_string(const std::string &key)
{
	return root.get(key, "").asString();
}

float* JSONHelper::get_vector(const std::string &key, unsigned int *size)
{
	assert(size != NULL);

	const Json::Value json_vector = root[key];

	*size = json_vector.size();
	
	if(json_vector.size() == 0) { return NULL; }

	float *vector = (float*)malloc(*size * sizeof(float));
	if(!vector) { return NULL; }

	for(int i = 0; i < json_vector.size(); i++)
	{
		vector[i] = (float)json_vector[i].asDouble();
	}

	return vector;
}
float* JSONHelper::get_matrix(const std::string &key, unsigned int *size)
{
	assert(size != NULL);

	const Json::Value json_matrix = root[key];

	*size = json_matrix.size();
	
	if(json_matrix.size() == 0) { return NULL; }
	for(int i = 0; i < json_matrix.size(); i++)
	{
		if(!json_matrix[i].isArray() || json_matrix[i].size() < json_matrix.size())
		{
			*size = 0;
			return NULL;
		}
	}

	float *matrix = (float*)malloc(*size * *size * sizeof(float));
	if(!matrix) { return NULL; }

	for(int row = 0; row < json_matrix.size(); row++)
	{
		for(int column = 0; column < json_matrix.size() && column < json_matrix[row].size(); column++)
		{
			matrix[row*json_matrix.size() + column] = (float)json_matrix[row][column].asDouble();
		}
	}

	return matrix;
}

OptionType JSONHelper::get_type()
{
	const std::string type_string = get_string("type");

	if(type_string == "european") { return European; }
	else if(type_string == "asian_geometric") { return Asian_Geometric; }
	else if(type_string == "asian_arithmetic") { return Asian_Arithmetic; }
	else if(type_string == "basket_geometric") { return Basket_Geometric; }
	else if(type_string == "basket_arithmetic") { return Basket_Arithmetic; }
	else { return European; }
}
OptionDirection JSONHelper::get_direction()
{
	const std::string direction_string = get_string("direction");

	if(direction_string == "call") { return Call; }
	else if(direction_string == "put") { return Put; }
	else { return Call; }
}
ControlVariate JSONHelper::get_control_variate()
{
	const std::string control_variate_string = get_string("control_variate");

	if(control_variate_string == "none") { return None; }
	else if(control_variate_string == "geometric") { return Geometric; }
	else if(control_variate_string == "geometric_adjusted_strike") { return Geometric_AdjustedStrike; }
	else { return None; }
}
