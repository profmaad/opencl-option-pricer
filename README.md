opencl-option-pricer
====================

A basic Monte Carlo option pricer, using OpenCL.

Dependencies
============

This project uses the following external components:
* [CMake](http://cmake.org/) >=2.8 for the build process.
* Any [OpenCL](https://www.khronos.org/opencl/) implementation.
* [COPRTHR](https://github.com/browndeer/coprthr) 1.6 including STDCL to handle the OpenCL end of things.
* (JsonCpp)[http://jsoncpp.sourceforge.net/] >= 0.5.0 for parsing input and outputting results as JSON.

Building
========

This project uses CMake for its build process. Please note that possible search paths for both COPRTHR and JsonCpp
are hardcoded in the CMakeLists.txt file, since neither project provides pkg-config files - at least on Arch Linux.
If CMake can't find their respective headers or libraries on your system, please add the required search paths to this file.

```shell
cd opencl-option-pricer
mkdir build
cd build
cmake ../
make
```

This should build two binaries: random_test and opencl_option_pricer.
random_test is a very basic test program to perform somewhat of a sanity check for the PRNG that is used on the OpenCL device.
opencl_option_pricer is the actual option pricer binary.

Usage
=====

To price an option, you need to supply the input parameters in JSON format on STDIN of opencl_option_pricer. The program reads
STDIN until EOF, so it can be necessary to close STDOUT of your supplying program to start the calculation.
It outputs the results in JSON format on STDOUT once it is done.
No progress reporting is available.

For examples of how to construct the input JSON for different options, have a look at the tests in the tests/ directory.
Please note that the "expected" block in these files is used solely for testing purposes and is not required for option pricing.

Copyright
=========

Copyright 2014 Maximilian Wolter (Prof. MAAD)
