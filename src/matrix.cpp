/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <cmath>

# include "matrix.h"

/* adapted from http://rosettacode.org/wiki/Cholesky_decomposition#C */
void cholesky_decomposition(unsigned int size, const float *input, float *output)
{
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j <= i; j++)
		{
			float s = 0;
			for(int k = 0; k < j; k++)
			{
				s += output[i * size + k] * output[j * size + k];
			}
			output[i * size + j] = (i == j) ?
				sqrt(input[i * size + i] - s) :
				(1.0 / output[j * size + j] * (input[i * size + j] - s));
		}
	}

	// this algorithm generates the lower decomposition, we need the upper decomposition - which is the transpose of the lower
	for(int row = 0; row < size; row++)
	{
		for(int column = size-1; column >= 0; column--)
		{
			if(column > row) { output[row*size + column] = output[column*size + row]; }
			else if(column < row) { output[row*size + column] = 0; }
		}
	}
}
