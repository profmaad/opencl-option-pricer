/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# ifndef OCLOP_TYPES_H
# define OCLOP_TYPES_H

enum OptionType
{
	European = 0,
	Asian_Geometric,
	Asian_Arithmetic,
	Basket_Geometric,
	Basket_Arithmetic,

	OptionType_SIZE
};
enum OptionDirection
{
	Call = 0,
	Put,

	OptionDirection_SIZE
};
enum ControlVariate
{
	None = 0,
	Geometric,
	Geometric_AdjustedStrike,

	ControlVariate_SIZE
};

# endif /*OCLOP_TYPES_H*/
