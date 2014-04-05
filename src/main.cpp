/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <iostream>

# include <stdcl.h>

int main(int argc, char **argv)
{
	stdcl_init();

	unsigned int problem_size = 16;

	/* use default contexts, if no GPU use CPU */
	CLCONTEXT* context = (stdgpu)? stdgpu : stdcpu;

	unsigned int devnum = 0;

	clopen(context, NULL, CLLD_NOW);
	
	cl_kernel kernel = clsym(context, NULL, "matvecmult_kern", 0);
	if (!kernel)
	{
		std::cerr << "error: kernel = " << kernel << std::endl;
		return 1;
	}
	
	/* allocate OpenCL device-sharable memory */
	cl_float* aa = (float*)clmalloc(context, problem_size*problem_size*sizeof(cl_float), 0);
	cl_float* b  = (float*)clmalloc(context, problem_size*sizeof(cl_float), 0);
	cl_float* c  = (float*)clmalloc(context, problem_size*sizeof(cl_float), 0);
	
	/* initialize vectors a[] and b[], zero c[] */
	int i,j; 
	for(i=0; i < problem_size; i++)
	{
		for(j=0; j < problem_size; j++)
		{
			aa[i*problem_size+j] = 1.1f*i*j;
		}
	}
	for(i=0; i < problem_size; i++)
	{
		b[i] = 2.2f*i;
	}
	for(i=0; i < problem_size; i++)
	{
		c[i] = 0.0f;
	}

	/* define the computational domain and workgroup size */
	clndrange_t index_range = clndrange_init1d(0, problem_size, 2);

	/* non-blocking sync vectors a and b to device memory (copy to GPU)*/
	clmsync(context, devnum, aa, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(context, devnum,  b, CL_MEM_DEVICE|CL_EVENT_NOWAIT);

	/* non-blocking fork of the OpenCL kernel to execute on the GPU */
	clforka(context, devnum, kernel, &index_range, CL_EVENT_NOWAIT, problem_size, aa, b, c);

	/* non-blocking sync vector c to host memory (copy back to host) */
	clmsync(context, devnum,  c, CL_MEM_HOST|CL_EVENT_NOWAIT);

	/* force execution of operations in command queue (non-blocking call) */
	clflush(context, devnum, 0);

	/* block on completion of operations in command queue */
	clwait(context, devnum, CL_ALL_EVENT);

	for(i=0; i < problem_size; i++) { printf("%d %f %f\n", i, b[i], c[i]); }

	clfree(aa);
	clfree(b);
	clfree(c);

	return 0;
}
