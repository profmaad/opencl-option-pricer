/* (c) 2014 Maximilian Gerhard Wolter */

# include <string>

# include <json/json.h>

class JSONOutputter
{
public:
	JSONOutputter(float mean, float confidence_interval_lower, float confidence_interval_upper);

	void set_number_of_paths(unsigned int number_of_paths);

	std::string output();

private:
	float mean;
	float confidence_interval_lower, confidence_interval_upper;
	unsigned int number_of_paths;
};
