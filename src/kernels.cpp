/* (c) 2014 Maximilian Gerhard Wolter */

# include <cstddef>

# include "types.h"
# include "kernels.h"

const char* get_kernel_sym(OptionType type, ControlVariate control_variate)
{
	if(type >= OptionType_SIZE || control_variate >= ControlVariate_SIZE) { return NULL; }
	else { return KERNEL_SYMBOLS[type][control_variate]; }
}
