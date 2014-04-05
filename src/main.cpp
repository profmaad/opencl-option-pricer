/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <iostream>

# include <cstdio>

# include <stdcl.h>

int main(int argc, char **argv)
{
	float start_price = 100.0;
	float strike_price = 100.0;
	float maturity = 3.0;
	float volatility = 0.3;
	float risk_free_rate = 0.05;

	stdcl_init();

	/* use default contexts, if no GPU use CPU */
	CLCONTEXT* context = (stdgpu)? stdgpu : stdcpu;

	unsigned int devnum = 0;

	clopen(context, NULL, CLLD_NOW);
	
	cl_kernel kernel = clsym(context, NULL, "european", 0);
	if (!kernel)
	{
		std::cerr << "error: kernel = " << kernel << std::endl;
		return 1;
	}
	
	/* allocate OpenCL device-sharable memory */
	cl_float* prices = (float*)clmalloc(context, 2*sizeof(cl_float), 0);
	
	for(int i=0; i < 2; i++)
	{
		prices[i] = 0.0f;
	}

	/* define the computational domain and workgroup size */
	clndrange_t index_range = clndrange_init1d(0, 1, 1);

	/* non-blocking fork of the OpenCL kernel to execute on the GPU */
	clforka(context, devnum, kernel, &index_range, CL_EVENT_NOWAIT, start_price, strike_price, maturity, volatility, risk_free_rate, prices);

	/* non-blocking sync vector c to host memory (copy back to host) */
	clmsync(context, devnum,  prices, CL_MEM_HOST|CL_EVENT_NOWAIT);

	/* force execution of operations in command queue (non-blocking call) */
	clflush(context, devnum, 0);

	/* block on completion of operations in command queue */
	clwait(context, devnum, CL_ALL_EVENT);

	std::cout << "Inputs:" << std::endl;
	printf("\tStart price:    %10.2f HKD\n", start_price);
	printf("\tStrike price:   %10.2f HKD\n", strike_price);
	printf("\tMaturity:       %10.3f years\n", maturity);
	printf("\tVolatility:     %10.5f %%\n", volatility*100);
	printf("\tRisk-free rate: %10.5f %%\n", risk_free_rate*100);

	std::cout << std::endl << std::endl << "Results:" << std::endl;
	printf("\tCall price:     %10.5f HKD\n", prices[0]);
	printf("\tPut price:      %10.5f HKD\n", prices[1]);

	clfree(prices);

	return 0;
}
