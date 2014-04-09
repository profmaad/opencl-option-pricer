/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_JSON_HELPER_HPP
# define OCLOP_JSON_HELPER_HPP

# include <string>

# include <json/json.h>

# include "types.h"

class JSONHelper
{
public:
	JSONHelper(const std::string &json_source);

	bool is_good();
	std::string get_parsing_errors();

	float get_float(const std::string &key);
	unsigned int get_uint(const std::string &key);
	const std::string get_string(const std::string &key);

	float* get_vector(const std::string &key, unsigned int *size);
	float* get_matrix(const std::string &key, unsigned int *size);
	
	OptionType get_type();
	OptionDirection get_direction();
	ControlVariate get_control_variate();

private:
	Json::Reader reader;
	Json::Value root;
	bool parsing_successful;
};

# endif /*OCLOP_JSON_HELPER_HPP*/
