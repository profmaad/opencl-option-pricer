/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_KERNELS_H
# define OCLOP_KERNELS_H

# include "types.h"

static const char *KERNEL_SYMBOLS[OptionType_SIZE][ControlVariate_SIZE] = { 
	{"european", "european", "european"},
	{"geometric_asian", "geometric_asian", "geometric_asian"},
	{"arithmetic_asian_no_cv", "arithmetic_asian_geometric_cv", "arithmetic_asian_geometric_cv"},
	{"geometric_basket", "geometric_basket", "geometric_basket"},
	{"arithmetic_basket_no_cv", "arithmetic_basket_geometric_cv", "arithmetic_basket_geometric_cv"}
};

const char* get_kernel_sym(OptionType type, ControlVariate control_variate);

# endif /*OCLOP_KERNELS_H*/
