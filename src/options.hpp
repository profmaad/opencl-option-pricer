/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_OPTIONS_HPP
# define OCLOP_OPTIONS_HPP

# include "option.hpp"
# include "opencl_option.hpp"

# include "closed_form_opencl_option.hpp"
# include "european_opencl_option.hpp"
# include "asian_geometric_opencl_option.hpp"
# include "basket_geometric_opencl_option.hpp"

# include "monte_carlo_opencl_option.hpp"
# include "asian_arithmetic_opencl_option.hpp"

# include "json_helper.hpp"

OpenCLOption* create_opencl_option(JSONHelper &parameters);

# endif /*OCLOP_OPTIONS_HPP*/
