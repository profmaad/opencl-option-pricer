/* (c) 2014 Maximilian Gerhard Wolter */

# ifndef OCLOP_OPTION_HPP
# define OCLOP_OPTION_HPP

class Option
{
public:
	virtual bool price(float *population_mean, float *confidence_interval_lower, float *confidence_interval_upper) = 0;
};

# endif /*OCLOP_OPTION_HPP*/
