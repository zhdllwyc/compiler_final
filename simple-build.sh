#!/bin/bash
pref=$1

SYCL_INCLUDE_PATH=/org/centers/cdgc/ComputeCpp/ComputeCpp-CE-2.2.1-x86_64-linux-gnu/include
OPENCL_INCLUDE_PATH=/org/centers/cdgc/ComputeCpp/OpenCL-Headers/
set -e
compute++ -sycl -sycl-target ptx64 -I$SYCL_INCLUDE_PATH -I$OPENCL_INCLUDE_PATH  $pref.cpp
g++ -include $pref.sycl $pref.cpp -I$SYCL_INCLUDE_PATH -I$OPENCL_INCLUDE_PATH -o $pref -Wl,-rpath,/org/centers/cdgc/ComputeCpp/ComputeCpp-CE-2.2.1-x86_64-linux-gnu/lib: /org/centers/cdgc/ComputeCpp/ComputeCpp-CE-2.2.1-x86_64-linux-gnu/lib/libComputeCpp.so /usr/lib64/libOpenCL.so
